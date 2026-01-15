import csv
import html
import io
import inspect
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import cv2
import pandas as pd
import plotly.graph_objects as go
import yaml

from core.args_schema import coerce_dict, default_cfg_dict
from core.model_zoo import ensure_model, is_model_cached, model_choices
from core.runner import build_command, start_process, stop_process, stream_logs, write_run_metadata
from ui.components import status_bar
from ui.predict import build_predict_tab
from ui.train import build_train_tab


ROOT = Path(__file__).resolve().parent
LAST_TRAIN_RUN_DIR: Optional[str] = None
LAST_RESULTS_CSV: Optional[str] = None
DEFAULT_CFG = default_cfg_dict()
MAX_LOG_LINES = 2000
ENABLE_DEBUG_MSG = False



def _load_css() -> str:
    css_path = ROOT / "ui" / "theme.css"
    return css_path.read_text(encoding="utf-8")


def _refresh_model_choices() -> Tuple[Dict[str, str], Dict[str, Dict]]:
    choices, meta_map = model_choices()
    return choices, meta_map


MODEL_CHOICES, MODEL_META = _refresh_model_choices()


def _model_key_from_choice(choice_value: Optional[str]) -> Optional[str]:
    if not choice_value:
        return None
    if choice_value in MODEL_META:
        return choice_value
    return MODEL_CHOICES.get(choice_value)


def _model_path_from_choice(choice_label: str) -> Optional[Path]:
    model_key = _model_key_from_choice(choice_label)
    if not model_key:
        return None
    meta = MODEL_META.get(model_key)
    if not meta:
        return None
    return Path("weights") / meta["release"] / meta["filename"]


def _model_hint(choice_label: Optional[str]) -> str:
    if not choice_label:
        return "Select a pretrained model to see download status."
    model_key = _model_key_from_choice(choice_label)
    if not model_key:
        return "Unknown model selection."
    meta = MODEL_META.get(model_key, {})
    path = Path("weights") / meta.get("release", "") / meta.get("filename", "")
    size = meta.get("size_mb")
    cached = False
    try:
        cached = is_model_cached(model_key)
    except Exception:
        cached = path.exists()
    if cached:
        return f"Cached at {path}"
    size_text = f"~{size} MB" if size else "unknown size"
    return f"Will download ({size_text}) to {path} when you start."


def _model_source_hint(source_kind: str, pretrained_label: Optional[str], local_path: Optional[str]) -> str:
    if source_kind == "Pretrained":
        return _model_hint(pretrained_label)
    if local_path:
        return f"Using local model: {local_path}"
    return "Provide a local .pt path."


def _resolve_model_path(
    source_kind: str,
    pretrained_label: Optional[str],
    local_path: Optional[str],
    progress=None,
    allow_download: bool = False,
) -> Path:
    if source_kind == "Pretrained":
        if not pretrained_label:
            raise ValueError("Please select a pretrained model.")
        model_key = _model_key_from_choice(pretrained_label)
        if not model_key:
            raise ValueError("Invalid pretrained model selection.")
        if allow_download:
            return ensure_model(model_key, progress=progress)
        expected = _model_path_from_choice(pretrained_label)
        if expected is None:
            raise ValueError("Unable to resolve pretrained model path.")
        return expected
    if not local_path:
        raise ValueError("Please provide a local model path.")
    return Path(local_path)


def _device_value(device_mode: str, single_gpu: Optional[str], multi_gpu: List[str]) -> Optional[str]:
    if device_mode == "auto":
        return None
    if device_mode == "cpu":
        return "cpu"
    if device_mode == "single":
        return single_gpu
    if device_mode == "multi":
        if not multi_gpu:
            return None
        return ",".join(multi_gpu)
    return None


def _normalize_file_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def _prepare_source(
    input_type: str,
    images,
    video,
    source_path: str,
    source_url: str,
    run_dir: Path,
    preview: bool = False,
) -> str:
    if input_type == "Images":
        image_list = images if isinstance(images, (list, tuple)) else ([images] if images else [])
        paths = [_normalize_file_path(p) for p in image_list]
        paths = [p for p in paths if p]
        if not paths:
            raise ValueError("Please upload at least one image.")
        if len(paths) == 1:
            return paths[0]
        if preview:
            return paths[0]
        source_dir = run_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, source_dir / Path(p).name)
        return str(source_dir)
    if input_type == "Video":
        path = _normalize_file_path(video)
        if not path:
            raise ValueError("Please upload a video.")
        return path
    if input_type == "Path":
        if not source_path:
            raise ValueError("Please provide a source path.")
        return source_path
    if input_type == "URL":
        if not source_url:
            raise ValueError("Please provide a source URL.")
        return source_url
    raise ValueError("Invalid source type.")


def _ffmpeg_path() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ensure_web_video(path: Path) -> Optional[Path]:
    target = path.with_name(path.stem + "_web.mp4")
    try:
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            return target
        ffmpeg = _ffmpeg_path()
        if ffmpeg:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(path),
                    "-movflags",
                    "+faststart",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and target.exists():
                return target
        if path.suffix.lower() in (".mp4", ".webm"):
            return path
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return path
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if width <= 0 or height <= 0:
            cap.release()
            return path
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return target if target.exists() else path
    except Exception:
        return path


def _collect_outputs(run_dir: Path) -> Tuple[List[str], Optional[str]]:
    images = []
    video = None
    for suffix in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        images.extend([str(p.resolve()) for p in run_dir.rglob(suffix)])
    for suffix in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        candidates = sorted(run_dir.rglob(suffix), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            candidate = candidates[0].resolve()
            web_video = _ensure_web_video(candidate)
            video = str(web_video.resolve()) if web_video else str(candidate)
            break
    return images, video


def _resolve_actual_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        return run_dir
    parent = run_dir.parent
    if not parent.exists():
        return run_dir
    candidates = []
    for candidate in parent.glob(f"{run_dir.name}*"):
        if any(candidate.rglob("*")):
            candidates.append(candidate)
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return run_dir


def _update_cli_preview(task: str, mode: str, args: Dict) -> str:
    _, preview = build_command(task, mode, args)
    return preview


def _run_dir_path(mode: str, name: Optional[str], create: bool) -> Path:
    name = (name or "").strip()
    if name:
        run_dir = Path("runs") / mode / name
    else:
        run_dir = Path("runs") / mode / "preview"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_local_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _run_dir_has_content(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    try:
        return any(run_dir.rglob("*"))
    except OSError:
        return False


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[error] {context}: {exc}", file=sys.stderr)
    traceback.print_exc()


def _save_uploaded_model(file_obj) -> Tuple[str, str]:
    if not file_obj:
        return "", "No file selected."
    src_path = _normalize_file_path(file_obj)
    if not src_path:
        return "", "Invalid upload."
    src = Path(src_path)
    if src.suffix.lower() != ".pt":
        return "", "Only .pt files are supported."
    dest_dir = ROOT / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        return str(dest), f"Already exists: {dest}"
    shutil.copy(src, dest)
    return str(dest), f"Saved to {dest}"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _normalize_csv_key(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _trim_log(log: str, max_lines: int = MAX_LOG_LINES) -> str:
    lines = log.splitlines()
    if len(lines) <= max_lines:
        return log
    kept = lines[-max_lines:]
    return "\n".join(kept) + "\n"


def _append_log(log: str, message: str, max_lines: int = MAX_LOG_LINES) -> str:
    if log and not log.endswith("\n"):
        log += "\n"
    log = f"{log}{message}\n"
    return _trim_log(log, max_lines=max_lines)


def _append_log_raw(log: str, text: str, max_lines: int = MAX_LOG_LINES) -> str:
    if not text:
        return log
    tokens = re.split(r"(\r|\n)", text)
    lines = log.splitlines()
    if not lines:
        lines = [""]
    if log.endswith("\n"):
        lines.append("")
    line_buf = lines[-1]
    overwrite_pending = False
    for token in tokens:
        if not token:
            continue
        if token == "\r":
            lines[-1] = line_buf
            line_buf = ""
            overwrite_pending = True
            continue
        if token == "\n":
            lines[-1] = line_buf
            lines.append("")
            line_buf = ""
            overwrite_pending = False
            continue
        if overwrite_pending:
            line_buf = token
            overwrite_pending = False
        else:
            line_buf += token
    if not (overwrite_pending and not line_buf):
        lines[-1] = line_buf
    log = "\n".join(lines)
    if log and not log.endswith("\n"):
        log += "\n"
    return _trim_log(log, max_lines=max_lines)


def _append_log_lines(log: str, text: str, max_lines: int = MAX_LOG_LINES) -> str:
    if not text:
        return log
    tokens = re.split(r"(\r|\n)", text)
    lines = log.splitlines()
    if not lines:
        lines = [""]
    if log.endswith("\n"):
        lines.append("")
    line_buf = lines[-1]
    overwrite_pending = False
    for token in tokens:
        if not token:
            continue
        if token == "\r":
            lines[-1] = line_buf
            line_buf = ""
            overwrite_pending = True
            continue
        if token == "\n":
            lines[-1] = line_buf
            lines.append("")
            line_buf = ""
            overwrite_pending = False
            continue
        if overwrite_pending:
            line_buf = token
            overwrite_pending = False
        else:
            line_buf += token
    if not (overwrite_pending and not line_buf):
        lines[-1] = line_buf
    log = "\n".join(lines)
    if log and not log.endswith("\n"):
        log += "\n"
    return _trim_log(log, max_lines=max_lines)


def _latest_log_line(current: str, text: str) -> str:
    if not text:
        return current or ""
    tokens = re.split(r"(\r|\n)", text)
    line_buf = current or ""
    overwrite_pending = False
    for token in tokens:
        if not token:
            continue
        if token == "\r":
            line_buf = ""
            overwrite_pending = True
            continue
        if token == "\n":
            line_buf = ""
            overwrite_pending = False
            continue
        if overwrite_pending:
            line_buf = token
            overwrite_pending = False
        else:
            line_buf += token
    return line_buf


def _is_progress_log(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"^\d+/\d+", stripped):
        return True
    if "%" in stripped and "it/s" in stripped:
        return True
    return False


def _select_ui_log_line(current: str, text: str) -> str:
    candidate = _latest_log_line(current, text)
    if not candidate.strip():
        return current
    if ENABLE_DEBUG_MSG:
        return candidate
    if _is_progress_log(candidate):
        return candidate
    return current


def _render_train_status(
    stage: str,
    data_path: str,
    epoch_current: Optional[int],
    epoch_total: Optional[int],
    metrics: Optional[List[Tuple[str, Optional[float]]]] = None,
) -> str:
    safe_stage = html.escape(stage or "Idle")
    safe_data = html.escape(data_path or "Not set")
    if epoch_current is not None and epoch_total:
        pct = max(0, min(int(epoch_current / epoch_total * 100), 100))
        epoch_text = f"{epoch_current}/{epoch_total}"
    else:
        pct = 0
        epoch_text = "-"
    metrics_html = ""
    if metrics:
        items = []
        for label, value in metrics:
            safe_label = html.escape(label)
            value_text = f"{value:.4f}" if value is not None else "-"
            items.append(
                f"<div style='min-width:120px;'>"
                f"<div style='color:#64748b;font-size:12px;'>{safe_label}</div>"
                f"<div style='font-weight:600;'>{value_text}</div>"
                f"</div>"
            )
        metrics_html = (
            "<div style='display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;'>"
            + "".join(items)
            + "</div>"
        )
    return f"""
    <div class="train-status-card" style="border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;">
      <div style="font-weight:600;margin-bottom:6px;">Training Status</div>
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <div><span style="color:#64748b;">Stage:</span> {safe_stage}</div>
        <div><span style="color:#64748b;">Data:</span> {safe_data}</div>
        <div><span style="color:#64748b;">Epoch:</span> {epoch_text}</div>
      </div>
      <div style="margin-top:8px;background:#e2e8f0;border-radius:999px;height:8px;overflow:hidden;">
        <div style="height:100%;background:#10b981;width:{pct}%;"></div>
      </div>
      {metrics_html}
    </div>
    """



class _ProgressTracker:
    def __init__(self, gr_progress, min_interval: float = 0.8):
        self.gr_progress = gr_progress
        self.min_interval = min_interval
        self.last_desc = None
        self.last_emitted = None
        self.last_emit = 0.0
        self.pending = None
        self.lock = threading.Lock()

    def __call__(self, pct: float, desc: str = "") -> None:
        if self.gr_progress is not None:
            self.gr_progress(pct, desc=desc)
        if not desc:
            return
        now = time.time()
        with self.lock:
            self.last_desc = desc
            if desc != self.last_emitted and now - self.last_emit >= self.min_interval:
                self.pending = desc

    def consume(self) -> Optional[str]:
        with self.lock:
            if not self.pending:
                return None
            msg = self.pending
            self.pending = None
            self.last_emitted = msg
            self.last_emit = time.time()
            return msg

    def flush(self) -> Optional[str]:
        with self.lock:
            if self.last_desc and self.last_desc != self.last_emitted:
                self.last_emitted = self.last_desc
                self.last_emit = time.time()
                return self.last_desc
            return None


def _format_size_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def _format_speed_mb(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    return f"{num_bytes / (1024 ** 2) / seconds:.2f} MB/s"


def _model_cache_state(pretrained_label: Optional[str]) -> Tuple[Optional[Path], bool, Optional[float]]:
    expected = _model_path_from_choice(pretrained_label or "")
    if expected is None:
        return None, False, None
    if not expected.exists():
        return expected, False, None
    try:
        return expected, True, expected.stat().st_mtime
    except OSError:
        return expected, True, None


def _extract_results_dir(log_text: str) -> Optional[Path]:
    for line in reversed(log_text.splitlines()):
        if "Results saved to" in line:
            cleaned = line.split("Results saved to", 1)[-1].strip()
            cleaned = cleaned.strip().strip(".")
            if cleaned:
                return _resolve_local_path(cleaned)
    return None


def _extract_save_dir(log_text: str) -> Optional[Path]:
    if not log_text:
        return None
    match = re.search(r"save_dir=([^\s]+)", log_text)
    if match:
        return _resolve_local_path(match.group(1))
    return _extract_results_dir(log_text)


def _find_results_csv(base_dir: Path) -> Optional[Path]:
    if base_dir.exists():
        direct = base_dir / "results.csv"
        if direct.exists():
            return direct
        candidates = list(base_dir.rglob("results.csv"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    parent = base_dir.parent
    if parent.exists():
        candidates = []
        for candidate in parent.glob(f"{base_dir.name}*"):
            csv_path = candidate / "results.csv"
            if csv_path.exists():
                candidates.append(csv_path)
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _tail_results_line(csv_path: Path, last_line: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not csv_path.exists():
        return last_line, None
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            rows = [line.strip() for line in f if line.strip()]
        if len(rows) < 2:
            return last_line, None
        tail = rows[-1]
        if tail == last_line:
            return last_line, None
        return tail, tail
    except Exception:
        return last_line, None


def _predict_status_snapshot(log: str) -> Tuple[str, str]:
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    last_line = lines[-1] if lines else "Idle."
    stage = "Idle"

    for line in reversed(lines):
        if "Download complete" in line:
            stage = "Download complete"
            break
        if "Downloading model" in line:
            stage = "Downloading model"
            break
        if "Found cached model file" in line or "Model cached" in line:
            stage = "Model cached"
            break
        if "Preparing source" in line:
            stage = "Preparing source"
            break
        if "Launching process" in line:
            stage = "Launching process"
            break
        if "Model running" in line:
            stage = "Running inference"
            break

    return stage, last_line


def _render_predict_status(log: str) -> str:
    stage, last_line = _predict_status_snapshot(log)
    safe_stage = html.escape(stage)
    safe_line = html.escape(last_line)
    return f"""
    <div class="predict-status">
      <div class="predict-status-header">
        <div class="predict-status-title">Status <span class="predict-status-hint">(Detailed logs in terminal)</span></div>
        <div class="predict-status-stage">{safe_stage}</div>
      </div>
      <div class="predict-status-body">
        <div class="predict-status-label">Last message</div>
        <div class="predict-status-value">{safe_line}</div>
      </div>
    </div>
    """


def _render_data_status(path: str) -> str:
    if not path:
        return "<div class='metrics-empty'>Missing data path.</div>"
    data_path = Path(path)
    if not data_path.exists():
        return "<div class='metrics-empty'>File not found.</div>"
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return f"<div class='metrics-empty'>Invalid YAML: {html.escape(str(exc))}</div>"

    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()

    def _resolve_split(value) -> List[Path]:
        if not value:
            return []
        if isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        paths = []
        for item in items:
            p = Path(item)
            if not p.is_absolute():
                p = (root / p).resolve()
            paths.append(p)
        return paths

    def _count_images(folder: Path) -> Optional[int]:
        if not folder.exists():
            return None
        if folder.is_file():
            return 1
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)

    splits = {
        "Train": _resolve_split(data.get("train")),
        "Val": _resolve_split(data.get("val")),
        "Test": _resolve_split(data.get("test")),
    }

    cards = []
    for name, paths in splits.items():
        if not paths:
            continue
        for p in paths:
            count = _count_images(p)
            count_text = f"{count} images" if count is not None else "missing"
            status = "Ready" if count else "Missing"
            cards.append(
                f"""
                <div class="status-card">
                  <div class="status-label">{html.escape(name)}</div>
                  <div class="status-value">{html.escape(str(p))}</div>
                  <div class="status-sub">{html.escape(count_text)} · {status}</div>
                </div>
                """
            )

    if not cards:
        return "<div class='metrics-empty'>No dataset splits found in YAML.</div>"

    return f"<div class='dataset-cards' style='display:flex;flex-direction:column;gap:10px;'>{''.join(cards)}</div>"


def _list_train_runs() -> List[str]:
    root = Path("runs") / "train"
    if not root.exists():
        return []
    runs = [str(p) for p in root.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: Path(p).stat().st_mtime, reverse=True)


def _init_results_state() -> Dict[str, Any]:
    return {"path": None, "offset": 0, "header": None, "rows": []}


def _read_results_incremental(state: Dict[str, Any], csv_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if not csv_path.exists():
        return state, pd.DataFrame()
    path = str(csv_path.resolve())
    if state.get("path") != path:
        state = _init_results_state()
        state["path"] = path

    try:
        file_size = csv_path.stat().st_size
    except OSError:
        return state, pd.DataFrame(state.get("rows", []))

    if file_size < state.get("offset", 0):
        state = _init_results_state()
        state["path"] = path

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        if not state.get("header"):
            header_line = f.readline()
            if not header_line:
                return state, pd.DataFrame(state.get("rows", []))
            header_raw = next(csv.reader([header_line.strip()]), [])
            if header_raw:
                header_raw[0] = header_raw[0].lstrip("\ufeff")
            state["header"] = [_normalize_csv_key(h) for h in header_raw]
            state["offset"] = f.tell()
        f.seek(state.get("offset", 0))
        new_text = f.read()
        state["offset"] = f.tell()

    if new_text:
        reader = csv.DictReader(io.StringIO(new_text), fieldnames=state.get("header") or [])
        rows = state.get("rows", [])
        for row in reader:
            cleaned = {_normalize_csv_key(str(k)): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if not any(cleaned.values()):
                continue
            rows.append(cleaned)
        state["rows"] = rows[-5000:]
    elif not state.get("rows") and file_size > 0:
        try:
            df_full = pd.read_csv(csv_path, encoding="utf-8", engine="python", skipinitialspace=True)
            df_full.rename(columns=lambda c: _normalize_csv_key(str(c)), inplace=True)
            state["header"] = [_normalize_csv_key(str(c)) for c in df_full.columns]
            state["rows"] = df_full.tail(5000).to_dict(orient="records")
            state["offset"] = file_size
        except Exception:
            return state, pd.DataFrame(state.get("rows", []))

    df = pd.DataFrame(state.get("rows", []))
    if not df.empty:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return state, df


def _find_col(df: pd.DataFrame, base: str) -> Optional[str]:
    if base in df.columns:
        return base
    if base.endswith("(B)"):
        alt = base.replace("(B)", "")
        return alt if alt in df.columns else None
    alt = f"{base}(B)"
    return alt if alt in df.columns else None


def _apply_view(df: pd.DataFrame, view_range: Optional[float]) -> pd.DataFrame:
    if df.empty:
        return df
    if view_range is None:
        return df
    try:
        count = int(view_range)
    except (TypeError, ValueError):
        return df
    if count < 0:
        return df
    if count == 0:
        return df.head(0)
    return df.tail(count)


def _apply_smoothing(df: pd.DataFrame, cols: List[str], window: int, enabled: bool) -> pd.DataFrame:
    if not enabled or window <= 1:
        return df
    smoothed = df.copy()
    for col in cols:
        if col in smoothed.columns:
            smoothed[col] = smoothed[col].rolling(window, min_periods=1).mean()
    return smoothed


def _plot_series(df: pd.DataFrame, x_col: str, series: List[Tuple[str, str]]) -> go.Figure:
    fig = go.Figure()
    x = df[x_col] if x_col in df.columns else pd.Series(range(len(df)))
    x = pd.to_numeric(x, errors="coerce")
    has_trace = False
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]

    for idx, (col, label) in enumerate(series):
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().sum() == 0:
            continue
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line=dict(color=color, width=2, shape="spline", smoothing=0.6),
                hovertemplate="<b>%{fullData.name}</b><br>%{y:.4f}<extra></extra>",
            )
        )
        has_trace = True

    if not has_trace:
        fig.add_annotation(
            text="No data yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#a4acb9"),
        )

    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
        font=dict(color="#e6e9ef"),
        margin=dict(l=32, r=20, t=32, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        xaxis=dict(
            title="epoch",
            hoverformat="EPOCH=%d",
            showgrid=True,
            gridcolor="rgba(42,51,66,0.28)",
            zeroline=False,
            tickfont=dict(color="#a4acb9"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(42,51,66,0.28)",
            zeroline=False,
            tickfont=dict(color="#a4acb9"),
        ),
    )
    return fig


def _empty_plot(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#a4acb9"),
    )
    fig.update_layout(
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
        font=dict(color="#e6e9ef"),
        margin=dict(l=32, r=20, t=32, b=28),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _render_kpis(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div class='metrics-empty'>No results yet.</div>"

    def _value(col_name: str) -> Optional[float]:
        col = _find_col(df, col_name)
        if col and col in df.columns:
            value = pd.to_numeric(df[col].iloc[-1], errors="coerce")
            return float(value) if pd.notna(value) else None
        return None

    def _best(col_name: str) -> Optional[float]:
        col = _find_col(df, col_name)
        if col and col in df.columns:
            value = pd.to_numeric(df[col], errors="coerce").max()
            return float(value) if pd.notna(value) else None
        return None

    def _trend(col_name: str) -> str:
        col = _find_col(df, col_name)
        if not col or col not in df.columns or len(df) < 2:
            return ""
        prev = pd.to_numeric(df[col].iloc[-2], errors="coerce")
        cur = pd.to_numeric(df[col].iloc[-1], errors="coerce")
        if pd.isna(prev) or pd.isna(cur):
            return ""
        arrow = "^" if cur >= prev else "v"
        return f"<span class='kpi-trend'>{arrow}</span>"

    epoch = _value("epoch")
    time_elapsed = _value("time")
    map5095 = _value("metrics/mAP50-95(B)")
    map50 = _value("metrics/mAP50(B)")
    precision = _value("metrics/precision(B)")
    recall = _value("metrics/recall(B)")
    lr_pg0 = _value("lr/pg0")
    val_loss = None
    val_candidates = [
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/box_om",
        "val/cls_om",
        "val/dfl_om",
        "val/box_oo",
        "val/cls_oo",
        "val/dfl_oo",
    ]
    for col in val_candidates:
        col_name = _find_col(df, col) or (col if col in df.columns else None)
        if col_name:
            val_loss = (val_loss or 0.0) + float(_value(col_name) or 0.0)

    def _card(label: str, value: str, trend: str = "") -> str:
        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{html.escape(label)}</div>
          <div class="kpi-value">{value}{trend}</div>
        </div>
        """

    cards = []
    if epoch is not None:
        cards.append(_card("Epoch", f"{int(epoch)}"))
    if time_elapsed is not None:
        cards.append(_card("Elapsed", f"{time_elapsed:.1f}"))
    if map5095 is not None:
        best = _best("metrics/mAP50-95(B)")
        suffix = f" / {best:.3f}" if best is not None else ""
        cards.append(_card("mAP50-95", f"{map5095:.3f}{suffix}", _trend("metrics/mAP50-95(B)")))
    if map50 is not None:
        cards.append(_card("mAP50", f"{map50:.3f}", _trend("metrics/mAP50(B)")))
    if precision is not None:
        cards.append(_card("Precision", f"{precision:.3f}", _trend("metrics/precision(B)")))
    if recall is not None:
        cards.append(_card("Recall", f"{recall:.3f}", _trend("metrics/recall(B)")))
    if lr_pg0 is not None:
        cards.append(_card("LR (pg0)", f"{lr_pg0:.6f}"))
    if val_loss is not None:
        cards.append(_card("Val Loss", f"{val_loss:.4f}"))

    return f"<div class='kpi-grid'>{''.join(cards)}</div>"


def _summarize_results(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None
    epoch_current = len(df)
    metrics: List[Tuple[str, Optional[float]]] = []
    for label, key in (
        ("mAP50-95", "metrics/mAP50-95(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("precision", "metrics/precision(B)"),
        ("recall", "metrics/recall(B)"),
        ("box loss", "train/box_loss"),
    ):
        col = _find_col(df, key)
        value = None
        if col and col in df.columns:
            num = pd.to_numeric(df[col].iloc[-1], errors="coerce")
            if pd.notna(num):
                value = float(num)
        metrics.append((label, value))
    return {"epoch_current": epoch_current, "metrics": metrics}


def _render_diag(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 6:
        return ""
    val_candidates = [
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/box_om",
        "val/cls_om",
        "val/dfl_om",
        "val/box_oo",
        "val/cls_oo",
        "val/dfl_oo",
    ]
    val_cols = []
    for col in val_candidates:
        col_name = _find_col(df, col) or (col if col in df.columns else None)
        if col_name:
            val_cols.append(col_name)
    map_col = _find_col(df, "metrics/mAP50-95(B)")
    if not val_cols or not map_col:
        return ""
    val_df = df[val_cols].apply(pd.to_numeric, errors="coerce")
    val_sum = val_df.sum(axis=1)
    recent = val_sum.tail(5)
    map_recent = pd.to_numeric(df[map_col], errors="coerce").tail(8)
    if recent.is_monotonic_increasing and map_recent.max() <= map_recent.iloc[0] + 1e-6:
        return "<div class='diag-note'>! Val loss rising while mAP stagnates. Consider lowering LR or adding augmentation.</div>"
    return ""


def _filter_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty:
        return df
    q = (query or "").strip()
    if q.startswith("cols="):
        tokens = [t.strip().lower() for t in q[5:].split(",") if t.strip()]
        if tokens:
            cols = [c for c in df.columns if any(t in c.lower() for t in tokens)]
            df = df[cols] if cols else df
        return df
    if q:
        mask = df.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False))
        df = df[mask.any(axis=1)]
    return df


def main() -> gr.Blocks:
    css = _load_css()
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            """
            <div class="app-hero">
              <h1>YOLOv10 WebUI v1.0.0</h1>
              <div class="subtle">Train and Predict with a fast, reproducible workflow.</div>
            </div>
            """
        )
        status_bar()
        mode_switch = gr.Radio(
            ["Basic", "Advanced"],
            value="Basic",
            label="Mode (Basic for quick starts, Advanced for full control.)",
        )

        with gr.Tabs():
            train_ui = build_train_tab()
            predict_ui = build_predict_tab()
        train_pretrained_state = gr.State(value=None)
        train_adv_pretrained_state = gr.State(value=None)
        predict_pretrained_state = gr.State(value=None)
        predict_adv_pretrained_state = gr.State(value=None)
        train_metrics_state = gr.State(value=_init_results_state())
        train_run_state = gr.State(value="")

        def _sync_mode(mode_value):
            return (
                gr.update(visible=mode_value == "Basic"),
                gr.update(visible=mode_value == "Advanced"),
                gr.update(visible=mode_value == "Basic"),
                gr.update(visible=mode_value == "Advanced"),
            )

        mode_switch.change(
            _sync_mode,
            inputs=mode_switch,
            outputs=[
                train_ui["basic_group"],
                train_ui["advanced_group"],
                predict_ui["basic_group"],
                predict_ui["advanced_group"],
            ],
        )

        def _validate_data(path: str) -> str:
            return _render_data_status(path)

        train_ui["validate_btn"].click(
            _validate_data,
            inputs=train_ui["data_path"],
            outputs=train_ui["data_status"],
        )

        def _resolve_run_path(run_state: str, output_dir: str, adv_output_dir: str) -> Optional[Path]:
            raw = (run_state or "").strip() or (output_dir or "").strip() or (adv_output_dir or "").strip()
            if not raw and LAST_TRAIN_RUN_DIR:
                raw = LAST_TRAIN_RUN_DIR
            if not raw and LAST_RESULTS_CSV:
                raw = str(Path(LAST_RESULTS_CSV).parent)
            if raw:
                return _resolve_actual_run_dir(_resolve_local_path(raw))
            return None

        def _update_metrics(
            run_state,
            output_dir,
            adv_output_dir,
            log_text,
            adv_log_text,
            view_range,
            table_filter,
            state,
        ):
            def _debug_metrics(message: str) -> None:
                if not ENABLE_DEBUG_MSG:
                    return
                last = state.get("debug_last")
                if message != last:
                    print(f"[metrics-debug] {message}", file=sys.stderr, flush=True)
                    state["debug_last"] = message

            resolved = _resolve_run_path(run_state, output_dir, adv_output_dir)
            extracted = _extract_save_dir(adv_log_text or log_text)
            if not resolved and LAST_RESULTS_CSV:
                _debug_metrics(f"fallback last_results_csv={LAST_RESULTS_CSV}")
            if not resolved and extracted:
                resolved = _resolve_actual_run_dir(extracted)
            if not resolved:
                _debug_metrics(
                    f"no_run resolved=None run_state={run_state!r} output_dir={output_dir!r} "
                    f"adv_output_dir={adv_output_dir!r} extracted={str(extracted) if extracted else None} "
                    f"last_run={LAST_TRAIN_RUN_DIR!r} last_csv={LAST_RESULTS_CSV!r}"
                )
                state["last_summary"] = None
                empty_fig = _empty_plot("Select or start a run")
                return (
                    empty_fig,
                    empty_fig,
                    empty_fig,
                    gr.update(value=pd.DataFrame()),
                    state,
                    run_state,
                )
            csv_path = _find_results_csv(resolved)
            if not csv_path:
                _debug_metrics(
                    f"no_results_csv resolved={str(resolved)} exists={resolved.exists()} "
                    f"last_csv={LAST_RESULTS_CSV!r}"
                )
                state["last_summary"] = None
                empty_fig = _empty_plot("Waiting for results.csv...")
                return (
                    empty_fig,
                    empty_fig,
                    empty_fig,
                    pd.DataFrame(),
                    state,
                    str(resolved),
                )
            if ENABLE_DEBUG_MSG:
                print(f"[metrics-debug] csv_path={csv_path}", file=sys.stderr, flush=True)
            resolved = csv_path.parent
            if state.get("debug_path") != str(resolved):
                _debug_metrics(f"resolved_path={str(resolved)} csv_path={str(csv_path)}")
                state["debug_path"] = str(resolved)
            try:
                state, df = _read_results_incremental(state, csv_path)
            except Exception as exc:
                _debug_metrics(f"read_results_error path={str(csv_path)} err={exc}")
                traceback.print_exc()
                empty_fig = _empty_plot("Waiting for results.csv...")
                return (
                    empty_fig,
                    empty_fig,
                    empty_fig,
                    pd.DataFrame(),
                    state,
                    str(resolved),
                )
            if df.empty:
                file_size = csv_path.stat().st_size if csv_path.exists() else 0
                _debug_metrics(f"results_empty path={str(csv_path)} size={file_size}")
                state["last_summary"] = None
                empty_fig = _empty_plot("Waiting for results.csv...")
                return (
                    empty_fig,
                    empty_fig,
                    empty_fig,
                    pd.DataFrame(),
                    state,
                    str(resolved),
                )
            if state.get("last_rows") == len(df):
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state,
                    run_state,
                )
            state["last_rows"] = len(df)
            state["last_summary"] = _summarize_results(df)
            cols_signature = tuple(df.columns)
            rows_count = len(df)
            if state.get("debug_rows") != rows_count or state.get("debug_cols") != cols_signature:
                _debug_metrics(f"results_ok rows={rows_count} cols={cols_signature}")
                state["debug_rows"] = rows_count
                state["debug_cols"] = cols_signature

            df_view = _apply_view(df, view_range)
            all_cols = [
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "train/box_om",
                "train/cls_om",
                "train/dfl_om",
                "train/box_oo",
                "train/cls_oo",
                "train/dfl_oo",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "val/box_om",
                "val/cls_om",
                "val/dfl_om",
                "val/box_oo",
                "val/cls_oo",
                "val/dfl_oo",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "lr/pg0",
                "lr/pg1",
                "lr/pg2",
            ]
            smooth_cols = [c for c in all_cols if _find_col(df_view, c)]
            df_smooth = _apply_smoothing(df_view, smooth_cols, 5, True)

            loss_series = []
            loss_groups = [
                ("train", "box"),
                ("train", "cls"),
                ("train", "dfl"),
                ("val", "box"),
                ("val", "cls"),
                ("val", "dfl"),
            ]
            for split, kind in loss_groups:
                base = f"{split}/{kind}_loss"
                col = _find_col(df_smooth, base)
                if col:
                    loss_series.append((col, f"{split} {kind} loss"))
                    continue
                for suffix, label in (("om", "om"), ("oo", "oo")):
                    alt = f"{split}/{kind}_{suffix}"
                    if alt in df_smooth.columns:
                        loss_series.append((alt, f"{split} {kind} {label}"))
            metric_series = []
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
                col = _find_col(df_smooth, key)
                if col:
                    metric_series.append((col, key.replace("metrics/", "")))
            lr_series = []
            for key in ("lr/pg0", "lr/pg1", "lr/pg2"):
                col = _find_col(df_smooth, key)
                if col:
                    lr_series.append((col, key))

            if not df_smooth.empty:
                last_row = df_smooth.tail(1).to_dict(orient="records")[0]
                if ENABLE_DEBUG_MSG:
                    print(
                        f"[metrics-debug] plot_input rows={len(df_smooth)} cols={list(df_smooth.columns)} last={last_row}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        f"[metrics-debug] loss_series={loss_series} metric_series={metric_series} lr_series={lr_series}",
                        file=sys.stderr,
                        flush=True,
                    )
            loss_fig = _plot_series(df_smooth, "epoch", loss_series)
            metric_fig = _plot_series(df_smooth, "epoch", metric_series)
            lr_fig = _plot_series(df_smooth, "epoch", lr_series)

            table_df = _filter_table(df_view, table_filter)
            table_signature = None
            if not table_df.empty:
                last_row = table_df.tail(1).to_numpy().ravel().tolist()
                table_signature = (len(table_df), tuple(table_df.columns), tuple(last_row))
            else:
                table_signature = (0, tuple(table_df.columns), tuple())
            if state.get("table_signature") == table_signature:
                table_update = gr.update()
            else:
                state["table_signature"] = table_signature
                table_update = table_df

            return (
                loss_fig,
                metric_fig,
                lr_fig,
                table_update,
                state,
                str(resolved),
            )

        load_kwargs = {}
        try:
            if "queue" in inspect.signature(demo.load).parameters:
                load_kwargs["queue"] = False
        except (TypeError, ValueError):
            pass

        demo.load(
            _update_metrics,
            inputs=[
                train_run_state,
                train_ui["output_dir"],
                train_ui["adv_output_dir"],
                train_ui["log_box"],
                train_ui["adv_log"],
                train_ui["view_range"],
                train_ui["table_filter"],
                train_metrics_state,
            ],
            outputs=[
                train_ui["loss_plot"],
                train_ui["metric_plot"],
                train_ui["lr_plot"],
                train_ui["metrics_table"],
                train_metrics_state,
                train_run_state,
            ],
            **load_kwargs,
        )

        train_ui["metrics_refresh"].click(
            _update_metrics,
            inputs=[
                train_run_state,
                train_ui["output_dir"],
                train_ui["adv_output_dir"],
                train_ui["log_box"],
                train_ui["adv_log"],
                train_ui["view_range"],
                train_ui["table_filter"],
                train_metrics_state,
            ],
            outputs=[
                train_ui["loss_plot"],
                train_ui["metric_plot"],
                train_ui["lr_plot"],
                train_ui["metrics_table"],
                train_metrics_state,
                train_run_state,
            ],
            queue=False,
        )

        metrics_timer = train_ui.get("metrics_timer")
        if metrics_timer is not None and hasattr(metrics_timer, "tick"):
            metrics_timer.tick(
                _update_metrics,
                inputs=[
                    train_run_state,
                    train_ui["output_dir"],
                    train_ui["adv_output_dir"],
                    train_ui["log_box"],
                    train_ui["adv_log"],
                    train_ui["view_range"],
                    train_ui["table_filter"],
                    train_metrics_state,
                ],
                outputs=[
                    train_ui["loss_plot"],
                    train_ui["metric_plot"],
                    train_ui["lr_plot"],
                    train_ui["metrics_table"],
                    train_metrics_state,
                    train_run_state,
                ],
                queue=False,
            )

        def _refresh_dropdowns(train_value=None, adv_train_value=None, pred_value=None, adv_pred_value=None):
            global MODEL_CHOICES, MODEL_META
            MODEL_CHOICES, MODEL_META = _refresh_model_choices()
            labels = [(label, key) for label, key in MODEL_CHOICES.items()]

            def _normalize(value: Optional[str]) -> Optional[str]:
                key = _model_key_from_choice(value)
                return key if key in MODEL_META else None

            return (
                gr.update(choices=labels, value=_normalize(train_value)),
                gr.update(choices=labels, value=_normalize(adv_train_value)),
                gr.update(choices=labels, value=_normalize(pred_value)),
                gr.update(choices=labels, value=_normalize(adv_pred_value)),
            )

        demo.load(
            _refresh_dropdowns,
            inputs=[
                train_ui["pretrained_model"],
                train_ui["adv_pretrained_model"],
                predict_ui["pretrained_model"],
                predict_ui["adv_pretrained_model"],
            ],
            outputs=[
                train_ui["pretrained_model"],
                train_ui["adv_pretrained_model"],
                predict_ui["pretrained_model"],
                predict_ui["adv_pretrained_model"],
            ],
        )

        def _update_train_hint(source_kind, pretrained_label, local_path):
            return _model_source_hint(source_kind, pretrained_label, local_path)

        def _update_predict_hint(source_kind, pretrained_label, local_path):
            return _model_source_hint(source_kind, pretrained_label, local_path)

        def _keep_last_choice(value: Optional[str], last_value: Optional[str]):
            normalized = _model_key_from_choice(value)
            if normalized:
                return gr.update(value=normalized), normalized
            if last_value:
                return gr.update(value=last_value), last_value
            return gr.update(value=value), last_value

        train_ui["model_source"].change(
            _update_train_hint,
            inputs=[train_ui["model_source"], train_ui["pretrained_model"], train_ui["local_model"]],
            outputs=train_ui["pretrained_hint"],
        )
        train_ui["pretrained_model"].change(
            _keep_last_choice,
            inputs=[train_ui["pretrained_model"], train_pretrained_state],
            outputs=[train_ui["pretrained_model"], train_pretrained_state],
        )
        train_ui["pretrained_model"].change(
            _update_train_hint,
            inputs=[train_ui["model_source"], train_ui["pretrained_model"], train_ui["local_model"]],
            outputs=train_ui["pretrained_hint"],
        )
        train_ui["local_model"].change(
            _update_train_hint,
            inputs=[train_ui["model_source"], train_ui["pretrained_model"], train_ui["local_model"]],
            outputs=train_ui["pretrained_hint"],
        )

        train_ui["adv_model_source"].change(
            _update_train_hint,
            inputs=[train_ui["adv_model_source"], train_ui["adv_pretrained_model"], train_ui["adv_local_model"]],
            outputs=train_ui["adv_pretrained_hint"],
        )
        train_ui["adv_pretrained_model"].change(
            _keep_last_choice,
            inputs=[train_ui["adv_pretrained_model"], train_adv_pretrained_state],
            outputs=[train_ui["adv_pretrained_model"], train_adv_pretrained_state],
        )
        train_ui["adv_pretrained_model"].change(
            _update_train_hint,
            inputs=[train_ui["adv_model_source"], train_ui["adv_pretrained_model"], train_ui["adv_local_model"]],
            outputs=train_ui["adv_pretrained_hint"],
        )
        train_ui["adv_local_model"].change(
            _update_train_hint,
            inputs=[train_ui["adv_model_source"], train_ui["adv_pretrained_model"], train_ui["adv_local_model"]],
            outputs=train_ui["adv_pretrained_hint"],
        )

        def _handle_upload(file_obj):
            path, status = _save_uploaded_model(file_obj)
            return (
                gr.update(value=path),
                gr.update(value="Local .pt"),
                gr.update(value=status),
            )

        train_ui["local_upload"].change(
            _handle_upload,
            inputs=train_ui["local_upload"],
            outputs=[train_ui["local_model"], train_ui["model_source"], train_ui["upload_status"]],
        )

        train_ui["adv_local_upload"].change(
            _handle_upload,
            inputs=train_ui["adv_local_upload"],
            outputs=[train_ui["adv_local_model"], train_ui["adv_model_source"], train_ui["adv_upload_status"]],
        )

        predict_ui["model_source"].change(
            _update_predict_hint,
            inputs=[predict_ui["model_source"], predict_ui["pretrained_model"], predict_ui["local_model"]],
            outputs=predict_ui["pretrained_hint"],
        )
        predict_ui["pretrained_model"].change(
            _keep_last_choice,
            inputs=[predict_ui["pretrained_model"], predict_pretrained_state],
            outputs=[predict_ui["pretrained_model"], predict_pretrained_state],
        )
        predict_ui["pretrained_model"].change(
            _update_predict_hint,
            inputs=[predict_ui["model_source"], predict_ui["pretrained_model"], predict_ui["local_model"]],
            outputs=predict_ui["pretrained_hint"],
        )
        predict_ui["local_model"].change(
            _update_predict_hint,
            inputs=[predict_ui["model_source"], predict_ui["pretrained_model"], predict_ui["local_model"]],
            outputs=predict_ui["pretrained_hint"],
        )

        predict_ui["adv_model_source"].change(
            _update_predict_hint,
            inputs=[predict_ui["adv_model_source"], predict_ui["adv_pretrained_model"], predict_ui["adv_local_model"]],
            outputs=predict_ui["adv_pretrained_hint"],
        )
        predict_ui["adv_pretrained_model"].change(
            _keep_last_choice,
            inputs=[predict_ui["adv_pretrained_model"], predict_adv_pretrained_state],
            outputs=[predict_ui["adv_pretrained_model"], predict_adv_pretrained_state],
        )
        predict_ui["adv_pretrained_model"].change(
            _update_predict_hint,
            inputs=[predict_ui["adv_model_source"], predict_ui["adv_pretrained_model"], predict_ui["adv_local_model"]],
            outputs=predict_ui["adv_pretrained_hint"],
        )
        predict_ui["adv_local_model"].change(
            _update_predict_hint,
            inputs=[predict_ui["adv_model_source"], predict_ui["adv_pretrained_model"], predict_ui["adv_local_model"]],
            outputs=predict_ui["adv_pretrained_hint"],
        )

        predict_ui["local_upload"].change(
            _handle_upload,
            inputs=predict_ui["local_upload"],
            outputs=[predict_ui["local_model"], predict_ui["model_source"], predict_ui["upload_status"]],
        )

        predict_ui["adv_local_upload"].change(
            _handle_upload,
            inputs=predict_ui["adv_local_upload"],
            outputs=[predict_ui["adv_local_model"], predict_ui["adv_model_source"], predict_ui["adv_upload_status"]],
        )

        def _basic_train_args(
            data_path,
            model_source,
            pretrained_model,
            local_model,
            epochs,
            patience,
            imgsz,
            batch,
            device_mode,
            single_gpu,
            multi_gpu,
            workers,
            run_name,
            create_run_dir: bool,
        ) -> Dict:
            model_path = _resolve_model_path(model_source, pretrained_model, local_model, allow_download=False)
            batch_value = -1 if batch == "auto" else int(batch)
            device = _device_value(device_mode, single_gpu, multi_gpu)
            args = {
                "data": data_path,
                "model": str(model_path),
                "epochs": int(epochs),
                "patience": int(patience),
                "imgsz": int(imgsz),
                "batch": batch_value,
                "device": device,
                "workers": int(workers),
                "amp": False,
                "verbose": True,
            }
            run_dir = _run_dir_path("train", run_name, create=create_run_dir)
            args["project"] = str(run_dir.parent)
            args["name"] = run_dir.name
            args["exist_ok"] = True
            return args

        def _update_basic_train_cli(*inputs):
            try:
                args = _basic_train_args(*inputs, create_run_dir=False)
                return _update_cli_preview("detect", "train", args)
            except Exception as exc:
                return f"Error: {exc}"

        basic_train_inputs = [
            train_ui["data_path"],
            train_ui["model_source"],
            train_ui["pretrained_model"],
            train_ui["local_model"],
            train_ui["epochs"],
            train_ui["patience"],
            train_ui["imgsz"],
            train_ui["batch"],
            train_ui["device_mode"],
            train_ui["single_gpu"],
            train_ui["multi_gpu"],
            train_ui["workers"],
            train_ui["run_name"],
        ]

        for comp in basic_train_inputs:
            comp.change(_update_basic_train_cli, inputs=basic_train_inputs, outputs=train_ui["cli_preview_basic"])

        def _run_basic_train(
            data_path,
            model_source,
            pretrained_model,
            local_model,
            epochs,
            patience,
            imgsz,
            batch,
            device_mode,
            single_gpu,
            multi_gpu,
            workers,
            run_name,
            view_range,
            table_filter,
            metrics_state,
            progress=gr.Progress(),
        ):
            try:
                total_epochs = int(epochs)
            except (TypeError, ValueError):
                total_epochs = None
            best_display = ""
            last_display = ""
            status_html = _render_train_status("Idle", data_path, None, total_epochs)
            status_stage = "Idle"
            run_dir = _run_dir_path("train", run_name, create=False)
            run_dir_str = str(run_dir)
            run_dir_abs = str(_resolve_local_path(run_dir_str))
            try:
                global LAST_TRAIN_RUN_DIR
                LAST_TRAIN_RUN_DIR = run_dir_abs
                metrics_state = metrics_state or {}

                def _emit_update(
                    status_html_value,
                    output_dir_value,
                    best_path_value,
                    last_path_value,
                    conflict_group_value,
                    conflict_message_value,
                    run_state_value,
                    metrics_state_value,
                ):
                    (
                        loss_fig,
                        metric_fig,
                        lr_fig,
                        table_update,
                        metrics_state_value,
                        run_state_value,
                    ) = _update_metrics(
                        run_state_value,
                        output_dir_value,
                        "",
                        "",
                        "",
                        view_range,
                        table_filter,
                        metrics_state_value,
                    )
                    summary = metrics_state_value.get("last_summary") if isinstance(metrics_state_value, dict) else None
                    if summary:
                        status_html_value = _render_train_status(
                            "Training",
                            data_path,
                            summary.get("epoch_current"),
                            total_epochs,
                            summary.get("metrics"),
                        )
                    return (
                        status_html_value,
                        output_dir_value,
                        best_path_value,
                        last_path_value,
                        conflict_group_value,
                        conflict_message_value,
                        run_state_value,
                        loss_fig,
                        metric_fig,
                        lr_fig,
                        table_update,
                        metrics_state_value,
                    )

                status_stage = "Resolving model"
                status_line = "[status] Resolving model..."
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
                args = _basic_train_args(
                    data_path,
                    model_source,
                    pretrained_model,
                    local_model,
                    epochs,
                    patience,
                    imgsz,
                    batch,
                    device_mode,
                    single_gpu,
                    multi_gpu,
                    workers,
                    run_name,
                    create_run_dir=False,
                )
                run_dir = Path(args["project"]) / args["name"]
                run_dir_str = str(run_dir)
                run_dir_abs = str(_resolve_local_path(run_dir_str))
                LAST_TRAIN_RUN_DIR = run_dir_abs
                if _run_dir_has_content(run_dir):
                    conflict_group, conflict_message = _show_conflict(run_dir)
                    update = _emit_update(
                        status_html,
                        "",
                        best_display,
                        last_display,
                        conflict_group,
                        conflict_message,
                        "",
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                    return
                run_dir.mkdir(parents=True, exist_ok=True)
                expected_path, existed, mtime_before = (None, False, None)
                if model_source == "Pretrained":
                    expected_path, existed, mtime_before = _model_cache_state(pretrained_model)
                if expected_path:
                    if existed:
                        status_stage = "Verifying model"
                        status_line = "[status] Found cached model file, verifying checksum..."
                    else:
                        status_stage = "Downloading model"
                        status_line = "[status] Downloading model from GitHub release..."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                start_time = time.time()
                if model_source == "Pretrained":
                    tracker = _ProgressTracker(progress)
                    result: Dict[str, Path] = {}
                    error: Dict[str, Exception] = {}

                    def worker():
                        try:
                            result["path"] = _resolve_model_path(
                                model_source,
                                pretrained_model,
                                local_model,
                                progress=tracker,
                                allow_download=True,
                            )
                        except Exception as exc:
                            error["exc"] = exc

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    while thread.is_alive():
                        msg = tracker.consume()
                        if msg:
                            status_stage = "Downloading model"
                            status_line = f"[download] {msg}"
                            print(status_line, flush=True)
                            status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                            update = _emit_update(
                                status_html,
                                run_dir_str,
                                best_display,
                                last_display,
                                gr.update(visible=False),
                                "",
                                run_dir_abs,
                                metrics_state,
                            )
                            metrics_state = update[-1]
                            run_dir_abs = update[6]
                            yield update
                        time.sleep(0.2)
                    thread.join()
                    msg = tracker.flush()
                    if msg:
                        status_stage = "Downloading model"
                        status_line = f"[download] {msg}"
                        print(status_line, flush=True)
                        status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                        update = _emit_update(
                            status_html,
                            run_dir_str,
                            best_display,
                            last_display,
                            gr.update(visible=False),
                            "",
                            run_dir_abs,
                            metrics_state,
                        )
                        metrics_state = update[-1]
                        run_dir_abs = update[6]
                        yield update
                    if error.get("exc"):
                        raise error["exc"]
                    model_path = result["path"]
                else:
                    model_path = _resolve_model_path(
                        model_source,
                        pretrained_model,
                        local_model,
                        progress=progress,
                        allow_download=True,
                    )
                elapsed = time.time() - start_time
                if expected_path and not existed:
                    size_bytes = model_path.stat().st_size if model_path.exists() else 0
                    speed = _format_speed_mb(size_bytes, elapsed)
                    status_stage = "Model ready"
                    status_line = f"[status] Download complete: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed})."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                elif expected_path and existed:
                    mtime_after = model_path.stat().st_mtime if model_path.exists() else None
                    if mtime_before and mtime_after and mtime_after > mtime_before:
                        size_bytes = model_path.stat().st_size if model_path.exists() else 0
                        speed = _format_speed_mb(size_bytes, elapsed)
                        status_stage = "Model updated"
                        status_line = (
                            f"[status] Model updated after checksum check: {_format_size_mb(size_bytes)} "
                            f"in {elapsed:.1f}s ({speed})."
                        )
                    else:
                        status_stage = "Model ready"
                        status_line = "[status] Model cached, skip download."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                args["model"] = str(model_path)
                cmd, preview = build_command("detect", "train", args)
                write_run_metadata(run_dir, args, preview)
                status_stage = "Starting training"
                status_line = "[status] Launching process..."
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
                process = start_process(cmd)
                if ENABLE_DEBUG_MSG:
                    print(f"[train-debug] run_dir={run_dir_str}", file=sys.stderr, flush=True)
                    print(
                        f"[train-debug] results_csv={_resolve_local_path(run_dir_str) / 'results.csv'}",
                        file=sys.stderr,
                        flush=True,
                    )
                results_csv_path = _resolve_local_path(run_dir_str) / "results.csv"
                results_csv_logged = False
                had_real_output = False
                for line in stream_logs(
                    process,
                    heartbeat=None,
                    heartbeat_message=None,
                ):
                    if line and not line.startswith("[status] Training running"):
                        had_real_output = True
                    if not results_csv_logged and results_csv_path.exists():
                        global LAST_RESULTS_CSV
                        LAST_RESULTS_CSV = str(results_csv_path)
                        weights_dir = results_csv_path.parent / "weights"
                        best_display = str((weights_dir / "best.pt").resolve())
                        last_display = str((weights_dir / "last.pt").resolve())
                        if ENABLE_DEBUG_MSG:
                            print(
                                f"[train-debug] results_csv_found={results_csv_path}",
                                file=sys.stderr,
                                flush=True,
                            )
                        results_csv_logged = True
                    print(line, end="" if line.endswith("\n") else "\n", flush=True)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
            except Exception as exc:
                _log_exception("basic train", exc)
                status_stage = "Error"
                status_line = f"Error: {exc}"
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    "",
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    "",
                    metrics_state,
                )
                yield update

        basic_train_start_inputs = basic_train_inputs + [
            train_ui["view_range"],
            train_ui["table_filter"],
            train_metrics_state,
        ]

        basic_train_event = train_ui["start_btn"].click(
            _run_basic_train,
            inputs=basic_train_start_inputs,
            outputs=[
                train_ui["log_box"],
                train_ui["output_dir"],
                train_ui["best_path"],
                train_ui["last_path"],
                train_ui["basic_conflict_group"],
                train_ui["basic_conflict_message"],
                train_run_state,
                train_ui["loss_plot"],
                train_ui["metric_plot"],
                train_ui["lr_plot"],
                train_ui["metrics_table"],
                train_metrics_state,
            ],
        )

        def _stop_train():
            stop_process()
            return _render_train_status("Stopped by user.", "", None, None)

        def _stop_predict():
            stop_process()
            return _render_predict_status("[status] Stopped by user.")

        train_ui["stop_btn"].click(
            _stop_train,
            inputs=None,
            outputs=train_ui["log_box"],
            cancels=[basic_train_event],
        )

        def _show_conflict(run_dir: Path) -> Tuple[Any, Any]:
            message = f"Run directory already exists: `{run_dir}`."
            return gr.update(visible=True), message

        def _hide_conflict() -> Tuple[Any, str]:
            return gr.update(visible=False), ""

        def _handle_conflict_action(action: str, new_name: str, run_name: str) -> Tuple[Any, Any, Any]:
            if action == "Rename run":
                updated = (new_name or "").strip() or run_name
                return gr.update(value=updated), gr.update(visible=False), ""
            return gr.update(value=run_name), gr.update(visible=False), ""

        train_ui["basic_conflict_confirm"].click(
            _handle_conflict_action,
            inputs=[train_ui["basic_conflict_action"], train_ui["basic_conflict_name"], train_ui["run_name"]],
            outputs=[train_ui["run_name"], train_ui["basic_conflict_group"], train_ui["basic_conflict_message"]],
        )
        train_ui["basic_conflict_cancel"].click(
            _hide_conflict,
            inputs=None,
            outputs=[train_ui["basic_conflict_group"], train_ui["basic_conflict_message"]],
        )

        train_ui["adv_conflict_confirm"].click(
            _handle_conflict_action,
            inputs=[train_ui["adv_conflict_action"], train_ui["adv_conflict_name"], train_ui["adv_run_name"]],
            outputs=[train_ui["adv_run_name"], train_ui["adv_conflict_group"], train_ui["adv_conflict_message"]],
        )
        train_ui["adv_conflict_cancel"].click(
            _hide_conflict,
            inputs=None,
            outputs=[train_ui["adv_conflict_group"], train_ui["adv_conflict_message"]],
        )

        def _set_predict_model(best_path, last_path):
            path = best_path or last_path
            if not path:
                return gr.update(value=""), gr.update(value="Local .pt")
            return gr.update(value=path), gr.update(value="Local .pt")

        def _advanced_train_args(
            adv_data_path,
            adv_model_source,
            adv_pretrained_model,
            adv_local_model,
            run_name,
            values: Dict[str, str],
            create_run_dir: bool,
        ) -> Dict:
            model_path = _resolve_model_path(adv_model_source, adv_pretrained_model, adv_local_model, allow_download=False)
            values = dict(values)
            if adv_data_path:
                values["data"] = adv_data_path
            values["model"] = str(model_path)
            run_dir = _run_dir_path("train", run_name, create=create_run_dir)
            values["project"] = str(run_dir.parent)
            values["name"] = run_dir.name
            values["exist_ok"] = True
            values["verbose"] = True
            return values

        def _gather_adv_values(adv_flat: Dict[str, Any], adv_values: List):
            keys = list(adv_flat.keys())
            values = dict(zip(keys, adv_values))
            coerced = coerce_dict(values, DEFAULT_CFG)
            coerced.pop("mode", None)
            coerced.pop("task", None)
            return coerced

        adv_components = list(train_ui["adv_flat"].values())

        def _update_adv_train_cli(
            adv_data_path,
            adv_model_source,
            adv_pretrained_model,
            adv_local_model,
            run_name,
            *adv_values,
        ):
            try:
                values = _gather_adv_values(train_ui["adv_flat"], list(adv_values))
                try:
                    total_epochs = int(values.get("epochs")) if values.get("epochs") is not None else None
                except (TypeError, ValueError):
                    total_epochs = None
                args = _advanced_train_args(
                    adv_data_path,
                    adv_model_source,
                    adv_pretrained_model,
                    adv_local_model,
                    run_name,
                    values,
                    False,
                )
                return _update_cli_preview("detect", "train", args)
            except Exception as exc:
                return f"Error: {exc}"

        adv_train_inputs = [
            train_ui["adv_data_path"],
            train_ui["adv_model_source"],
            train_ui["adv_pretrained_model"],
            train_ui["adv_local_model"],
            train_ui["adv_run_name"],
        ] + adv_components

        for comp in adv_train_inputs:
            comp.change(
                _update_adv_train_cli,
                inputs=adv_train_inputs,
                outputs=train_ui["cli_preview_adv"],
            )

        def _run_adv_train(
            adv_data_path,
            adv_model_source,
            adv_pretrained_model,
            adv_local_model,
            run_name,
            *adv_values,
            progress=gr.Progress(),
        ):
            total_epochs = None
            best_display = ""
            last_display = ""
            status_html = _render_train_status("Idle", adv_data_path, None, total_epochs)
            status_stage = "Idle"
            run_dir = _run_dir_path("train", run_name, create=False)
            run_dir_str = str(run_dir)
            run_dir_abs = str(_resolve_local_path(run_dir_str))
            try:
                if len(adv_values) < 3:
                    raise ValueError("Missing view/table inputs for training.")
                view_range, table_filter, metrics_state = adv_values[-3], adv_values[-2], adv_values[-1]
                adv_values = adv_values[:-3]
                global LAST_TRAIN_RUN_DIR
                LAST_TRAIN_RUN_DIR = run_dir_abs
                metrics_state = metrics_state or {}

                def _emit_update(
                    status_html_value,
                    adv_output_dir_value,
                    best_path_value,
                    last_path_value,
                    conflict_group_value,
                    conflict_message_value,
                    run_state_value,
                    metrics_state_value,
                ):
                    (
                        loss_fig,
                        metric_fig,
                        lr_fig,
                        table_update,
                        metrics_state_value,
                        run_state_value,
                    ) = _update_metrics(
                        run_state_value,
                        "",
                        adv_output_dir_value,
                        "",
                        "",
                        view_range,
                        table_filter,
                        metrics_state_value,
                    )
                    summary = metrics_state_value.get("last_summary") if isinstance(metrics_state_value, dict) else None
                    if summary:
                        status_html_value = _render_train_status(
                            "Training",
                            adv_data_path,
                            summary.get("epoch_current"),
                            total_epochs,
                            summary.get("metrics"),
                        )
                    return (
                        status_html_value,
                        adv_output_dir_value,
                        best_path_value,
                        last_path_value,
                        conflict_group_value,
                        conflict_message_value,
                        run_state_value,
                        loss_fig,
                        metric_fig,
                        lr_fig,
                        table_update,
                        metrics_state_value,
                    )

                status_stage = "Resolving model"
                status_line = "[status] Resolving model..."
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
                values = _gather_adv_values(train_ui["adv_flat"], list(adv_values))
                run_dir = _run_dir_path("train", run_name, create=False)
                run_dir_str = str(run_dir)
                run_dir_abs = str(_resolve_local_path(run_dir_str))
                LAST_TRAIN_RUN_DIR = run_dir_abs
                if _run_dir_has_content(run_dir):
                    conflict_group, conflict_message = _show_conflict(run_dir)
                    update = _emit_update(
                        status_html,
                        "",
                        best_display,
                        last_display,
                        conflict_group,
                        conflict_message,
                        "",
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                    return
                expected_path, existed, mtime_before = (None, False, None)
                if adv_model_source == "Pretrained":
                    expected_path, existed, mtime_before = _model_cache_state(adv_pretrained_model)
                if expected_path:
                    if existed:
                        status_stage = "Verifying model"
                        status_line = "[status] Found cached model file, verifying checksum..."
                    else:
                        status_stage = "Downloading model"
                        status_line = "[status] Downloading model from GitHub release..."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                start_time = time.time()
                if adv_model_source == "Pretrained":
                    tracker = _ProgressTracker(progress)
                    result: Dict[str, Path] = {}
                    error: Dict[str, Exception] = {}

                    def worker():
                        try:
                            result["path"] = _resolve_model_path(
                                adv_model_source,
                                adv_pretrained_model,
                                adv_local_model,
                                progress=tracker,
                                allow_download=True,
                            )
                        except Exception as exc:
                            error["exc"] = exc

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    while thread.is_alive():
                        msg = tracker.consume()
                        if msg:
                            status_stage = "Downloading model"
                            status_line = f"[download] {msg}"
                            print(status_line, flush=True)
                            status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                            update = _emit_update(
                                status_html,
                                run_dir_str,
                                best_display,
                                last_display,
                                gr.update(visible=False),
                                "",
                                run_dir_abs,
                                metrics_state,
                            )
                            metrics_state = update[-1]
                            run_dir_abs = update[6]
                            yield update
                        time.sleep(0.2)
                    thread.join()
                    msg = tracker.flush()
                    if msg:
                        status_stage = "Downloading model"
                        status_line = f"[download] {msg}"
                        print(status_line, flush=True)
                        status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                        update = _emit_update(
                            status_html,
                            run_dir_str,
                            best_display,
                            last_display,
                            gr.update(visible=False),
                            "",
                            run_dir_abs,
                            metrics_state,
                        )
                        metrics_state = update[-1]
                        run_dir_abs = update[6]
                        yield update
                    if error.get("exc"):
                        raise error["exc"]
                    model_path = result["path"]
                else:
                    model_path = _resolve_model_path(
                        adv_model_source,
                        adv_pretrained_model,
                        adv_local_model,
                        progress=progress,
                        allow_download=True,
                    )
                elapsed = time.time() - start_time
                if expected_path and not existed:
                    size_bytes = model_path.stat().st_size if model_path.exists() else 0
                    speed = _format_speed_mb(size_bytes, elapsed)
                    status_stage = "Model ready"
                    status_line = f"[status] Download complete: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed})."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                elif expected_path and existed:
                    mtime_after = model_path.stat().st_mtime if model_path.exists() else None
                    if mtime_before and mtime_after and mtime_after > mtime_before:
                        size_bytes = model_path.stat().st_size if model_path.exists() else 0
                        speed = _format_speed_mb(size_bytes, elapsed)
                        status_stage = "Model updated"
                        status_line = (
                            f"[status] Model updated after checksum check: {_format_size_mb(size_bytes)} "
                            f"in {elapsed:.1f}s ({speed})."
                        )
                    else:
                        status_stage = "Model ready"
                        status_line = "[status] Model cached, skip download."
                    print(status_line, flush=True)
                    status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                values["model"] = str(model_path)
                if adv_data_path:
                    values["data"] = adv_data_path
                run_dir.mkdir(parents=True, exist_ok=True)
                values["project"] = str(run_dir.parent)
                values["name"] = run_dir.name
                cmd, preview = build_command("detect", "train", values)
                write_run_metadata(run_dir, values, preview)
                status_stage = "Starting training"
                status_line = "[status] Launching process..."
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
                process = start_process(cmd)
                if ENABLE_DEBUG_MSG:
                    print(f"[train-debug] run_dir={run_dir_str}", file=sys.stderr, flush=True)
                    print(
                        f"[train-debug] results_csv={_resolve_local_path(run_dir_str) / 'results.csv'}",
                        file=sys.stderr,
                        flush=True,
                    )
                results_csv_path = _resolve_local_path(run_dir_str) / "results.csv"
                results_csv_logged = False
                had_real_output = False
                for line in stream_logs(
                    process,
                    heartbeat=None,
                    heartbeat_message=None,
                ):
                    if line and not line.startswith("[status] Training running"):
                        had_real_output = True
                    if not results_csv_logged and results_csv_path.exists():
                        global LAST_RESULTS_CSV
                        LAST_RESULTS_CSV = str(results_csv_path)
                        weights_dir = results_csv_path.parent / "weights"
                        best_display = str((weights_dir / "best.pt").resolve())
                        last_display = str((weights_dir / "last.pt").resolve())
                        if ENABLE_DEBUG_MSG:
                            print(
                                f"[train-debug] results_csv_found={results_csv_path}",
                                file=sys.stderr,
                                flush=True,
                            )
                        results_csv_logged = True
                    print(line, end="" if line.endswith("\n") else "\n", flush=True)
                    update = _emit_update(
                        status_html,
                        run_dir_str,
                        best_display,
                        last_display,
                        gr.update(visible=False),
                        "",
                        run_dir_abs,
                        metrics_state,
                    )
                    metrics_state = update[-1]
                    run_dir_abs = update[6]
                    yield update
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                update = _emit_update(
                    status_html,
                    run_dir_str,
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    run_dir_abs,
                    metrics_state,
                )
                metrics_state = update[-1]
                run_dir_abs = update[6]
                yield update
            except Exception as exc:
                _log_exception("advanced train", exc)
                status_stage = "Error"
                status_line = f"Error: {exc}"
                print(status_line, flush=True)
                status_html = _render_train_status(status_stage, adv_data_path, None, total_epochs)
                update = _emit_update(
                    status_html,
                    "",
                    best_display,
                    last_display,
                    gr.update(visible=False),
                    "",
                    "",
                    metrics_state,
                )
                yield update

        adv_train_start_inputs = adv_train_inputs + [
            train_ui["view_range"],
            train_ui["table_filter"],
            train_metrics_state,
        ]

        adv_train_event = train_ui["adv_start"].click(
            _run_adv_train,
            inputs=adv_train_start_inputs,
            outputs=[
                train_ui["adv_log"],
                train_ui["adv_output_dir"],
                train_ui["adv_best_path"],
                train_ui["adv_last_path"],
                train_ui["adv_conflict_group"],
                train_ui["adv_conflict_message"],
                train_run_state,
                train_ui["loss_plot"],
                train_ui["metric_plot"],
                train_ui["lr_plot"],
                train_ui["metrics_table"],
                train_metrics_state,
            ],
        )
        train_ui["adv_stop"].click(
            _stop_train,
            inputs=None,
            outputs=train_ui["adv_log"],
            cancels=[adv_train_event],
        )
        def _basic_predict_args(
            model_source,
            pretrained_model,
            local_model,
            input_type,
            images,
            video,
            source_path,
            source_url,
            conf,
            iou,
            imgsz,
            device_mode,
            single_gpu,
            multi_gpu,
        ) -> Dict:
            model_path = _resolve_model_path(model_source, pretrained_model, local_model, allow_download=False)
            device = _device_value(device_mode, single_gpu, multi_gpu)
            args = {
                "model": str(model_path),
                "conf": conf,
                "iou": iou,
                "imgsz": int(imgsz),
                "device": device,
                "save": True,
            }
            return args

        basic_predict_inputs = [
            predict_ui["model_source"],
            predict_ui["pretrained_model"],
            predict_ui["local_model"],
            predict_ui["input_type"],
            predict_ui["images"],
            predict_ui["video"],
            predict_ui["source_path"],
            predict_ui["source_url"],
            predict_ui["conf"],
            predict_ui["iou"],
            predict_ui["imgsz"],
            predict_ui["device_mode"],
            predict_ui["single_gpu"],
            predict_ui["multi_gpu"],
        ]

        def _update_basic_predict_cli(*inputs):
            try:
                args = _basic_predict_args(*inputs)
                run_dir = _run_dir_path("predict", None, create=False)
                source = _prepare_source(inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], run_dir, preview=True)
                args["source"] = source
                args["project"] = str(run_dir.parent)
                args["name"] = run_dir.name
                return _update_cli_preview("detect", "predict", args)
            except Exception as exc:
                return f"Error: {exc}"

        for comp in basic_predict_inputs:
            comp.change(_update_basic_predict_cli, inputs=basic_predict_inputs, outputs=predict_ui["cli_preview_basic"])

        def _run_basic_predict(*inputs, progress=gr.Progress()):
            log = ""
            try:
                log = _append_log(log, "[status] Resolving model...")
                print("[status] Resolving model...")
                yield _render_predict_status(log), [], None, ""
                args = _basic_predict_args(*inputs)
                expected_path, existed, mtime_before = (None, False, None)
                if inputs[0] == "Pretrained":
                    expected_path, existed, mtime_before = _model_cache_state(inputs[1])
                if expected_path:
                    if existed:
                        log = _append_log(log, "[status] Found cached model file, verifying checksum...")
                    else:
                        log = _append_log(log, "[status] Downloading model from GitHub release...")
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                start_time = time.time()
                if inputs[0] == "Pretrained":
                    tracker = _ProgressTracker(progress)
                    result: Dict[str, Path] = {}
                    error: Dict[str, Exception] = {}

                    def worker():
                        try:
                            result["path"] = _resolve_model_path(
                                inputs[0],
                                inputs[1],
                                inputs[2],
                                progress=tracker,
                                allow_download=True,
                            )
                        except Exception as exc:
                            error["exc"] = exc

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    while thread.is_alive():
                        msg = tracker.consume()
                        if msg:
                            log = _append_log(log, f"[download] {msg}")
                            print(f"[download] {msg}")
                            yield _render_predict_status(log), [], None, ""
                        time.sleep(0.2)
                    thread.join()
                    msg = tracker.flush()
                    if msg:
                        log = _append_log(log, f"[download] {msg}")
                        print(f"[download] {msg}")
                        yield _render_predict_status(log), [], None, ""
                    if error.get("exc"):
                        raise error["exc"]
                    model_path = result["path"]
                else:
                    model_path = _resolve_model_path(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        progress=progress,
                        allow_download=True,
                    )
                elapsed = time.time() - start_time
                if expected_path and not existed:
                    size_bytes = model_path.stat().st_size if model_path.exists() else 0
                    speed = _format_speed_mb(size_bytes, elapsed)
                    log = _append_log(
                        log,
                        f"[status] Download complete: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed}).",
                    )
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                elif expected_path and existed:
                    mtime_after = model_path.stat().st_mtime if model_path.exists() else None
                    if mtime_before and mtime_after and mtime_after > mtime_before:
                        size_bytes = model_path.stat().st_size if model_path.exists() else 0
                        speed = _format_speed_mb(size_bytes, elapsed)
                        log = _append_log(
                            log,
                            f"[status] Model updated after checksum check: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed}).",
                        )
                    else:
                        log = _append_log(log, "[status] Model cached, skip download.")
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                args["model"] = str(model_path)
                log = _append_log(log, f"[status] Using model: {model_path}")
                print(log.splitlines()[-1])
                run_dir = _run_dir_path("predict", None, create=True)
                yield _render_predict_status(log), [], None, str(run_dir)
                log = _append_log(log, "[status] Preparing source...")
                print(log.splitlines()[-1])
                source = _prepare_source(inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], run_dir, preview=False)
                args["source"] = source
                args["project"] = str(run_dir.parent)
                args["name"] = run_dir.name
                cmd, preview = build_command("detect", "predict", args)
                write_run_metadata(run_dir, args, preview)
                log = _append_log(log, "[status] Launching process...")
                print(log.splitlines()[-1])
                yield _render_predict_status(log), [], None, str(run_dir)
                process = start_process(cmd)
                for line in stream_logs(
                    process,
                    heartbeat=5.0,
                    heartbeat_message="[status] Model running, waiting for output...",
                ):
                    clean_line = _strip_ansi(line)
                    log = _append_log_lines(log, clean_line)
                    print(clean_line, end="" if clean_line.endswith("\n") else "\n")
                    yield _render_predict_status(log), [], None, str(run_dir)
                parsed_dir = _extract_results_dir(log)
                actual_dir = parsed_dir if parsed_dir else _resolve_actual_run_dir(run_dir)
                images, video = _collect_outputs(actual_dir)
                yield _render_predict_status(log), images, video, str(actual_dir.resolve())
            except Exception as exc:
                _log_exception("basic predict", exc)
                log = _append_log(log, f"Error: {exc}")
                print(f"[error] {exc}")
                yield _render_predict_status(log), [], None, ""

        predict_ui["start_btn"].click(
            _run_basic_predict,
            inputs=basic_predict_inputs,
            outputs=[
                predict_ui["status_html"],
                predict_ui["output_gallery"],
                predict_ui["output_video"],
                predict_ui["output_dir"],
            ],
        )

        predict_ui["stop_btn"].click(_stop_predict, inputs=None, outputs=predict_ui["status_html"])

        def _advanced_predict_args(
            adv_model_source,
            adv_pretrained_model,
            adv_local_model,
            adv_input_type,
            adv_images,
            adv_video,
            adv_source_path,
            adv_source_url,
            values: Dict,
            create_run_dir: bool,
        ) -> Dict:
            model_path = _resolve_model_path(adv_model_source, adv_pretrained_model, adv_local_model, allow_download=False)
            values = dict(values)
            values["model"] = str(model_path)
            run_dir = _run_dir_path("predict", None, create=create_run_dir)
            source = _prepare_source(
                adv_input_type,
                adv_images,
                adv_video,
                adv_source_path,
                adv_source_url,
                run_dir,
                preview=not create_run_dir,
            )
            values["source"] = source
            values["project"] = str(run_dir.parent)
            values["name"] = run_dir.name
            values["save"] = True
            return values

        adv_predict_components = list(predict_ui["adv_flat"].values())

        def _update_adv_predict_cli(adv_model_source, adv_pretrained_model, adv_local_model, adv_input_type, adv_images, adv_video, adv_source_path, adv_source_url, *adv_values):
            try:
                values = _gather_adv_values(predict_ui["adv_flat"], list(adv_values))
                args = _advanced_predict_args(
                    adv_model_source,
                    adv_pretrained_model,
                    adv_local_model,
                    adv_input_type,
                    adv_images,
                    adv_video,
                    adv_source_path,
                    adv_source_url,
                    values,
                    False,
                )
                return _update_cli_preview("detect", "predict", args)
            except Exception as exc:
                return f"Error: {exc}"

        adv_predict_inputs = [
            predict_ui["adv_model_source"],
            predict_ui["adv_pretrained_model"],
            predict_ui["adv_local_model"],
            predict_ui["adv_input_type"],
            predict_ui["adv_images"],
            predict_ui["adv_video"],
            predict_ui["adv_source_path"],
            predict_ui["adv_source_url"],
        ] + adv_predict_components

        for comp in adv_predict_inputs:
            comp.change(_update_adv_predict_cli, inputs=adv_predict_inputs, outputs=predict_ui["cli_preview_adv"])

        def _run_adv_predict(adv_model_source, adv_pretrained_model, adv_local_model, adv_input_type, adv_images, adv_video, adv_source_path, adv_source_url, *adv_values, progress=gr.Progress()):
            log = ""
            try:
                log = _append_log(log, "[status] Resolving model...")
                print("[status] Resolving model...")
                yield _render_predict_status(log), [], None, ""
                values = _gather_adv_values(predict_ui["adv_flat"], list(adv_values))
                expected_path, existed, mtime_before = (None, False, None)
                if adv_model_source == "Pretrained":
                    expected_path, existed, mtime_before = _model_cache_state(adv_pretrained_model)
                if expected_path:
                    if existed:
                        log = _append_log(log, "[status] Found cached model file, verifying checksum...")
                    else:
                        log = _append_log(log, "[status] Downloading model from GitHub release...")
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                start_time = time.time()
                if adv_model_source == "Pretrained":
                    tracker = _ProgressTracker(progress)
                    result: Dict[str, Path] = {}
                    error: Dict[str, Exception] = {}

                    def worker():
                        try:
                            result["path"] = _resolve_model_path(
                                adv_model_source,
                                adv_pretrained_model,
                                adv_local_model,
                                progress=tracker,
                                allow_download=True,
                            )
                        except Exception as exc:
                            error["exc"] = exc

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    while thread.is_alive():
                        msg = tracker.consume()
                        if msg:
                            log = _append_log(log, f"[download] {msg}")
                            print(f"[download] {msg}")
                            yield _render_predict_status(log), [], None, ""
                        time.sleep(0.2)
                    thread.join()
                    msg = tracker.flush()
                    if msg:
                        log = _append_log(log, f"[download] {msg}")
                        print(f"[download] {msg}")
                        yield _render_predict_status(log), [], None, ""
                    if error.get("exc"):
                        raise error["exc"]
                    model_path = result["path"]
                else:
                    model_path = _resolve_model_path(
                        adv_model_source,
                        adv_pretrained_model,
                        adv_local_model,
                        progress=progress,
                        allow_download=True,
                    )
                elapsed = time.time() - start_time
                if expected_path and not existed:
                    size_bytes = model_path.stat().st_size if model_path.exists() else 0
                    speed = _format_speed_mb(size_bytes, elapsed)
                    log = _append_log(
                        log,
                        f"[status] Download complete: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed}).",
                    )
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                elif expected_path and existed:
                    mtime_after = model_path.stat().st_mtime if model_path.exists() else None
                    if mtime_before and mtime_after and mtime_after > mtime_before:
                        size_bytes = model_path.stat().st_size if model_path.exists() else 0
                        speed = _format_speed_mb(size_bytes, elapsed)
                        log = _append_log(
                            log,
                            f"[status] Model updated after checksum check: {_format_size_mb(size_bytes)} in {elapsed:.1f}s ({speed}).",
                        )
                    else:
                        log = _append_log(log, "[status] Model cached, skip download.")
                    print(log.splitlines()[-1])
                    yield _render_predict_status(log), [], None, ""
                values["model"] = str(model_path)
                log = _append_log(log, f"[status] Using model: {model_path}")
                print(log.splitlines()[-1])
                run_dir = _run_dir_path("predict", None, create=True)
                yield _render_predict_status(log), [], None, str(run_dir)
                log = _append_log(log, "[status] Preparing source...")
                print(log.splitlines()[-1])
                source = _prepare_source(
                    adv_input_type,
                    adv_images,
                    adv_video,
                    adv_source_path,
                    adv_source_url,
                    run_dir,
                    preview=False,
                )
                values["source"] = source
                values["project"] = str(run_dir.parent)
                values["name"] = run_dir.name
                cmd, preview = build_command("detect", "predict", values)
                write_run_metadata(run_dir, values, preview)
                log = _append_log(log, "[status] Launching process...")
                print(log.splitlines()[-1])
                yield _render_predict_status(log), [], None, str(run_dir)
                process = start_process(cmd)
                for line in stream_logs(
                    process,
                    heartbeat=5.0,
                    heartbeat_message="[status] Model running, waiting for output...",
                ):
                    clean_line = _strip_ansi(line)
                    log = _append_log_lines(log, clean_line)
                    print(clean_line, end="" if clean_line.endswith("\n") else "\n")
                    yield _render_predict_status(log), [], None, str(run_dir)
                parsed_dir = _extract_results_dir(log)
                actual_dir = parsed_dir if parsed_dir else _resolve_actual_run_dir(run_dir)
                images, video = _collect_outputs(actual_dir)
                yield _render_predict_status(log), images, video, str(actual_dir.resolve())
            except Exception as exc:
                _log_exception("advanced predict", exc)
                log = _append_log(log, f"Error: {exc}")
                print(f"[error] {exc}")
                yield _render_predict_status(log), [], None, ""

        predict_ui["adv_start"].click(
            _run_adv_predict,
            inputs=adv_predict_inputs,
            outputs=[
                predict_ui["adv_status_html"],
                predict_ui["adv_gallery"],
                predict_ui["adv_video"],
                predict_ui["adv_output_dir"],
            ],
        )
        predict_ui["adv_stop"].click(_stop_predict, inputs=None, outputs=predict_ui["adv_status_html"])

        gr.HTML(
            "<div class='app-footer'>WebUI developed by <a href='https://github.com/LeoWang0814/yolov10-webui' target='_blank'>LeoWang</a></div>"
        )

    return demo


if __name__ == "__main__":
    main().queue(concurrency_count=2).launch()
