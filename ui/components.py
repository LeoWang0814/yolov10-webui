from typing import List, Tuple

import gradio as gr
import torch

from core.gpu import get_system_status


def status_bar() -> gr.HTML:
    status = get_system_status()
    html = f"""
    <div class="status-bar">
      <div class="pill">Torch: {status['torch']}</div>
      <div class="pill">CUDA: {status['cuda']} ({status['cuda_version']})</div>
      <div class="pill">GPU: {status['gpu_count']} {status['gpu_list']}</div>
      <div class="pill">Platform: {status['platform']}</div>
    </div>
    """
    return gr.HTML(html)


def gpu_ids() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def device_defaults() -> Tuple[str, List[str]]:
    if torch.cuda.is_available():
        return "auto", gpu_ids()
    return "cpu", []
