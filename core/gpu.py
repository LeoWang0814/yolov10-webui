import platform
from typing import Dict, List

import torch


def _format_gpus() -> List[str]:
    if not torch.cuda.is_available():
        return []
    gpus = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        gpus.append(f"{idx}: {props.name} ({total_gb:.1f} GB)")
    return gpus


def get_system_status() -> Dict[str, str]:
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    gpus = _format_gpus()
    return {
        "torch": torch.__version__,
        "cuda": "available" if cuda_available else "unavailable",
        "cuda_version": cuda_version or "N/A",
        "gpu_count": str(len(gpus)),
        "gpu_list": ", ".join(gpus) if gpus else "None",
        "platform": platform.platform(),
    }
