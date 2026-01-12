import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
ZOO_PATH = ROOT / "tools" / "download.json"


def _load_zoo() -> Dict:
    with ZOO_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_release_key(zoo: Dict) -> str:
    releases = zoo.get("releases", {})
    if not releases:
        raise ValueError("No releases found in download.json")
    return sorted(releases.keys())[-1]


def _weights_dir(release_key: str) -> Path:
    return ROOT / "weights" / release_key


def _hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _env_int(name: str, default: int, min_value: int = 1, max_value: int = 32) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _mirror_url(url: str, mirror: str) -> str:
    if not mirror:
        return url
    if mirror.endswith("/"):
        return mirror + url
    return mirror + "/" + url


def _source_urls(zoo: Dict, meta: Dict) -> List[str]:
    sources = meta.get("sources", {})
    if not sources:
        return []
    preferred = os.getenv("YOLOV10_DOWNLOAD_SOURCE") or os.getenv("YOLOV10_MODEL_SOURCE")
    default_source = zoo.get("default_source")
    urls: List[str] = []

    if preferred and preferred in sources:
        urls.append(sources[preferred])
    elif default_source in sources:
        urls.append(sources[default_source])
    else:
        urls.append(next(iter(sources.values())))

    for url in sources.values():
        if url not in urls:
            urls.append(url)

    mirror = os.getenv("YOLOV10_DOWNLOAD_MIRROR")
    if mirror:
        mirrored = [_mirror_url(url, mirror) for url in urls]
        urls = mirrored + urls

    return urls


def _probe_download(session: requests.Session, url: str) -> Tuple[int, bool]:
    try:
        with session.head(url, allow_redirects=True, timeout=(10, 30)) as resp:
            if resp.ok:
                total = int(resp.headers.get("content-length", 0) or 0)
                accept_ranges = resp.headers.get("accept-ranges", "")
                return total, "bytes" in accept_ranges.lower()
    except requests.RequestException:
        pass

    with session.get(url, stream=True, headers={"Range": "bytes=0-0"}, timeout=(10, 30)) as resp:
        resp.raise_for_status()
        content_range = resp.headers.get("content-range", "")
        if content_range and "/" in content_range:
            total = int(content_range.split("/")[-1])
            return total, True
        total = int(resp.headers.get("content-length", 0) or 0)
        return total, False


def _update_progress(
    progress,
    downloaded: int,
    total: int,
    start: float,
    last_update: float,
) -> float:
    now = time.time()
    if progress is None:
        return last_update
    if now - last_update < 0.4:
        return last_update
    if total:
        pct = downloaded / total
        speed_mb = (downloaded / max(now - start, 1e-6)) / (1024 ** 2)
        progress(pct, desc=f"Downloading model... {pct*100:.1f}% {speed_mb:.2f} MB/s")
    else:
        speed_mb = (downloaded / max(now - start, 1e-6)) / (1024 ** 2)
        progress(0, desc=f"Downloading model... {speed_mb:.2f} MB/s")
    return now


def _download_single(
    session: requests.Session,
    url: str,
    tmp_path: Path,
    total: int,
    progress=None,
) -> None:
    headers = {}
    downloaded = 0
    if tmp_path.exists() and total:
        existing = tmp_path.stat().st_size
        if 0 < existing < total:
            headers["Range"] = f"bytes={existing}-"
            downloaded = existing
        elif existing >= total:
            return
    start = time.time()
    last_update = start

    response = session.get(url, stream=True, headers=headers, timeout=(10, 30))
    with response:
        response.raise_for_status()
        mode = "ab" if headers else "wb"
        with tmp_path.open(mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                last_update = _update_progress(progress, downloaded, total, start, last_update)


def _download_parallel(
    session: requests.Session,
    url: str,
    tmp_path: Path,
    total: int,
    progress=None,
) -> None:
    min_part_size = _env_int("YOLOV10_DOWNLOAD_MIN_PART_MB", 8, min_value=1, max_value=128) * 1024 * 1024
    max_workers = _env_int("YOLOV10_DOWNLOAD_WORKERS", 4, min_value=1, max_value=8)
    workers = min(max_workers, max(1, total // min_part_size))
    if workers <= 1:
        _download_single(session, url, tmp_path, total, progress)
        return

    part_size = math.ceil(total / workers)
    part_paths = []
    ranges = []
    downloaded = 0

    for i in range(workers):
        start = i * part_size
        end = min(total - 1, (i + 1) * part_size - 1)
        part_path = tmp_path.with_suffix(tmp_path.suffix + f".part{i}")
        part_paths.append(part_path)
        ranges.append((start, end))
        if part_path.exists():
            downloaded += min(part_path.stat().st_size, end - start + 1)

    lock = Lock()
    start_time = time.time()
    last_update = start_time

    def worker(part_path: Path, start: int, end: int) -> None:
        nonlocal downloaded, last_update
        expected = end - start + 1
        offset = 0
        if part_path.exists():
            offset = part_path.stat().st_size
            if offset >= expected:
                return
        headers = {"Range": f"bytes={start + offset}-{end}"}
        response = session.get(url, stream=True, headers=headers, timeout=(10, 30))
        with response:
            if response.status_code == 200:
                raise ValueError("Server did not honor range requests.")
            response.raise_for_status()
            with part_path.open("ab") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    with lock:
                        downloaded += len(chunk)
                        last_update = _update_progress(progress, downloaded, total, start_time, last_update)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, part_path, start, end) for part_path, (start, end) in zip(part_paths, ranges)]
        for future in as_completed(futures):
            future.result()

    with tmp_path.open("wb") as out:
        for part_path in part_paths:
            with part_path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)

    for part_path in part_paths:
        part_path.unlink(missing_ok=True)

    if tmp_path.stat().st_size != total:
        raise ValueError("Downloaded size mismatch.")


def _cleanup_parts(tmp_path: Path) -> None:
    for part_path in tmp_path.parent.glob(tmp_path.name + ".part*"):
        part_path.unlink(missing_ok=True)


def _is_cached(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    actual = _hash_file(path)
    if actual != expected_sha256:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def model_choices() -> Tuple[Dict[str, str], Dict[str, Dict]]:
    zoo = _load_zoo()
    release_key = _latest_release_key(zoo)
    models = zoo["releases"][release_key]["models"]
    choices = {}
    meta_map = {}
    for key, meta in models.items():
        weights_dir = _weights_dir(release_key)
        weights_path = weights_dir / meta["filename"]
        cached = _is_cached(weights_path, meta["sha256"])
        label = f"{key} ({'cached' if cached else 'download'})"
        choices[label] = key
        meta_map[key] = {
            "release": release_key,
            "filename": meta["filename"],
            "sha256": meta["sha256"],
            "size_mb": meta.get("size_mb"),
            "url": meta["sources"]["github"],
        }
    return choices, meta_map


def model_path_for_key(model_key: str) -> Path:
    zoo = _load_zoo()
    release_key = _latest_release_key(zoo)
    models = zoo["releases"][release_key]["models"]
    if model_key not in models:
        raise ValueError(f"Unknown model key: {model_key}")
    meta = models[model_key]
    return _weights_dir(release_key) / meta["filename"]


def is_model_cached(model_key: str) -> bool:
    zoo = _load_zoo()
    release_key = _latest_release_key(zoo)
    models = zoo["releases"][release_key]["models"]
    if model_key not in models:
        raise ValueError(f"Unknown model key: {model_key}")
    meta = models[model_key]
    weights_path = _weights_dir(release_key) / meta["filename"]
    return _is_cached(weights_path, meta["sha256"])


def ensure_model(model_key: str, progress=None) -> Path:
    zoo = _load_zoo()
    release_key = _latest_release_key(zoo)
    models = zoo["releases"][release_key]["models"]
    if model_key not in models:
        raise ValueError(f"Unknown model key: {model_key}")
    meta = models[model_key]
    weights_dir = _weights_dir(release_key)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / meta["filename"]

    if _is_cached(weights_path, meta["sha256"]):
        return weights_path

    urls = _source_urls(zoo, meta)
    if not urls:
        raise ValueError("No download sources configured for this model.")
    tmp_path = weights_path.with_suffix(weights_path.suffix + ".tmp")
    session = _build_session()
    last_error = None
    for url in urls:
        try:
            total, accept_ranges = _probe_download(session, url)
            if tmp_path.exists() and not total:
                tmp_path.unlink()
            if accept_ranges and total:
                try:
                    _download_parallel(session, url, tmp_path, total, progress)
                except ValueError as exc:
                    if "range" in str(exc).lower():
                        _cleanup_parts(tmp_path)
                        _download_single(session, url, tmp_path, total, progress)
                    else:
                        raise
            else:
                _download_single(session, url, tmp_path, total, progress)
            if progress is not None:
                progress(1.0, desc="Verifying model checksum...")
            actual = _hash_file(tmp_path)
            if actual != meta["sha256"]:
                raise ValueError("SHA256 mismatch for downloaded model.")
            tmp_path.replace(weights_path)
            if progress is not None:
                progress(1.0, desc="Model cached.")
            return weights_path
        except Exception as exc:
            last_error = exc
            continue

    tmp_path.unlink(missing_ok=True)
    raise RuntimeError("Model download failed.") from last_error
