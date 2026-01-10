import platform
import re
import subprocess
from typing import Dict, List, Tuple

import torch


def _query_nvidia_smi() -> List[Dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, name, mem_total = parts
        gpus.append(
            {
                "index": idx,
                "name": name,
                "memory_gb": f"{float(mem_total) / 1024:.1f}",
            }
        )
    return gpus


def _query_nvidia_smi_summary() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    match = re.search(
        r"Driver Version:\s*([0-9.]+).*?CUDA Version:\s*([0-9.]+)",
        result.stdout,
        re.DOTALL,
    )
    if not match:
        return {}
    return {"driver_version": match.group(1), "driver_cuda": match.group(2)}


def _format_gpus() -> Tuple[List[Dict[str, str]], str]:
    if torch.cuda.is_available():
        gpus = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024 ** 3)
            gpus.append(
                {
                    "index": str(idx),
                    "name": props.name,
                    "memory_gb": f"{total_gb:.1f}",
                }
            )
        return gpus, "torch"
    gpus = _query_nvidia_smi()
    return gpus, "nvidia-smi" if gpus else "none"


def get_system_status() -> Dict[str, object]:
    cuda_available = torch.cuda.is_available()
    torch_cuda = torch.version.cuda or "N/A"
    gpus, gpu_source = _format_gpus()
    smi_summary = _query_nvidia_smi_summary()
    cuda_state = "available" if cuda_available else "unavailable"
    if not cuda_available and gpus:
        cuda_hint = "Install CUDA-enabled PyTorch to use the GPU."
        if smi_summary.get("driver_cuda"):
            cuda_note = f"Driver CUDA: {smi_summary['driver_cuda']}"
        else:
            cuda_note = "GPU driver detected."
    else:
        cuda_note = "OK" if cuda_available else "No CUDA runtime detected."
        cuda_hint = ""
    return {
        "torch": torch.__version__,
        "torch_cuda": torch_cuda,
        "cuda": cuda_state,
        "cuda_note": cuda_note,
        "cuda_hint": cuda_hint,
        "gpu_count": len(gpus),
        "gpu_list": gpus,
        "gpu_source": gpu_source,
        "driver_version": smi_summary.get("driver_version", ""),
        "driver_cuda": smi_summary.get("driver_cuda", ""),
        "platform": platform.platform(),
    }
