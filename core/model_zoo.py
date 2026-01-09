import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import requests


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

    url = meta["sources"]["github"]
    tmp_path = weights_path.with_suffix(weights_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        start = time.time()
        last_time = start
        last_bytes = 0
        last_print = start

        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if progress is not None and total:
                    delta_t = max(now - last_time, 1e-6)
                    speed = (downloaded - last_bytes) / delta_t
                    speed_mb = speed / (1024 ** 2)
                    pct = downloaded / total
                    progress(pct, desc=f"Downloading model... {pct*100:.1f}% {speed_mb:.2f} MB/s")
                    last_time = now
                    last_bytes = downloaded
                if now - last_print > 1.0:
                    if total:
                        pct = downloaded / total * 100
                        speed_mb = (downloaded / max(now - start, 1e-6)) / (1024 ** 2)
                        print(f"[download] {model_key}: {pct:.1f}% {speed_mb:.2f} MB/s")
                    else:
                        speed_mb = (downloaded / max(now - start, 1e-6)) / (1024 ** 2)
                        print(f"[download] {model_key}: {downloaded / (1024 ** 2):.1f} MB {speed_mb:.2f} MB/s")
                    last_print = now

        if progress is not None:
            progress(1.0, desc="Verifying model checksum...")
        actual = _hash_file(tmp_path)
        if actual != meta["sha256"]:
            raise ValueError("SHA256 mismatch for downloaded model.")

        tmp_path.replace(weights_path)
        if progress is not None:
            progress(1.0, desc="Model cached.")
        return weights_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
