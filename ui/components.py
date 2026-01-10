from typing import List, Tuple

import gradio as gr
import torch

from core.gpu import get_system_status


def status_bar() -> gr.HTML:
    status = get_system_status()
    hint_html = ""
    if status.get("cuda_hint"):
        hint_html = f"<div class='status-alert'>{status['cuda_hint']}</div>"
    gpu_cards = ""
    if status["gpu_list"]:
        for gpu in status["gpu_list"]:
            gpu_cards += (
                "<div class='gpu-item'>"
                f"<div class='gpu-name'>#{gpu['index']} {gpu['name']}</div>"
                f"<div class='gpu-meta'>{gpu['memory_gb']} GB</div>"
                "</div>"
            )
    else:
        gpu_cards = "<div class='gpu-empty'>No GPU detected</div>"
    source_note = ""
    if status["gpu_source"] != "none":
        source_note = f"Source: {status['gpu_source']}"
    html = f"""
    <div class="status-wrap">
      <div class="status-grid">
        <div class="status-card">
          <div class="status-label">Torch</div>
          <div class="status-value">{status['torch']}</div>
          <div class="status-sub">CUDA build: {status['torch_cuda']}</div>
        </div>
        <div class="status-card">
          <div class="status-label">CUDA</div>
          <div class="status-value">{status['cuda']}</div>
          <div class="status-sub">{status['cuda_note']}</div>
        </div>
        <div class="status-card status-platform">
          <div class="status-label">Platform</div>
          <div class="status-value">{status['platform']}</div>
        </div>
      </div>
      {hint_html}
      <div class="gpu-panel">
        <div class="gpu-header">
          <span>GPUs ({status['gpu_count']})</span>
          <span class="gpu-source">{source_note}</span>
        </div>
        <div class="gpu-list">
          {gpu_cards}
        </div>
      </div>
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
