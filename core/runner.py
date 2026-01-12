import json
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from core.gpu import get_system_status


_process = None
_log_queue: "queue.Queue[str]" = queue.Queue()
_stop_event = threading.Event()


def _reader_thread(process: subprocess.Popen) -> None:
    for line in iter(process.stdout.readline, ""):
        if _stop_event.is_set():
            break
        _log_queue.put(line)
    process.stdout.close()


def stop_process() -> None:
    global _process
    _stop_event.set()
    if _process and _process.poll() is None:
        _process.terminate()
    _process = None


def _run_dir(mode: str, name: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = name or f"{mode}-{timestamp}"
    run_dir = Path("runs") / mode / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_command(task: str, mode: str, args: Dict) -> Tuple[List[str], str]:
    cmd = ["yolo", task, mode]
    for key, value in args.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            cmd.append(f"{key}={str(value)}")
        elif isinstance(value, (list, tuple)):
            joined = ",".join(str(v) for v in value)
            cmd.append(f"{key}={joined}")
        else:
            cmd.append(f"{key}={value}")
    preview = " ".join([_quote_arg(x) for x in cmd])
    return cmd, preview


def _quote_arg(value: str) -> str:
    if " " in value or "\t" in value:
        return f"\"{value}\""
    return value


def write_run_metadata(run_dir: Path, args: Dict, command: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(args, indent=2), encoding="utf-8")
    (run_dir / "command.txt").write_text(command, encoding="utf-8")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "system": get_system_status(),
        "git": _git_commit(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def start_process(cmd: List[str]) -> subprocess.Popen:
    global _process
    if _process and _process.poll() is None:
        raise RuntimeError("Another process is already running.")
    _stop_event.clear()
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break
    _process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    thread = threading.Thread(target=_reader_thread, args=(_process,), daemon=True)
    thread.start()
    return _process


def stream_logs(
    process: subprocess.Popen,
    heartbeat: Optional[float] = None,
    heartbeat_message: Optional[str] = None,
) -> Iterable[str]:
    last_heartbeat = time.time()
    while process.poll() is None or not _log_queue.empty():
        if _stop_event.is_set():
            break
        try:
            line = _log_queue.get(timeout=0.2)
            yield line
            last_heartbeat = time.time()
        except queue.Empty:
            if heartbeat:
                now = time.time()
                if now - last_heartbeat >= heartbeat:
                    yield heartbeat_message or "[status] Running, waiting for output..."
                    last_heartbeat = now
            continue


def prepare_run(mode: str, name: Optional[str], args: Dict) -> Path:
    run_dir = _run_dir(mode, name)
    return run_dir
