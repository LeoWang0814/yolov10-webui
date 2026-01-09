from typing import Any, Dict, Tuple

import gradio as gr

from core.args_schema import build_grouped_defaults
from ui.components import device_defaults


def _input_for_default(key: str, default):
    label = f"{key} (default: {default})"
    if isinstance(default, bool):
        return gr.Checkbox(label=label, value=default)
    if isinstance(default, int):
        return gr.Number(label=label, value=default, precision=0)
    if isinstance(default, float):
        return gr.Number(label=label, value=default)
    return gr.Textbox(label=label, value="" if default is None else str(default))


def _build_advanced_inputs() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    grouped_defaults = build_grouped_defaults("predict")
    flat_components: Dict[str, Any] = {}
    grouped_components: Dict[str, Dict[str, Any]] = {}
    for group, items in grouped_defaults.items():
        grouped_components[group] = {}
        with gr.Accordion(group, open=False):
            for key, default in items.items():
                component = _input_for_default(key, default)
                flat_components[key] = component
                grouped_components[group][key] = component
    return flat_components, grouped_components


def build_predict_tab() -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    with gr.Tab("Predict"):
        with gr.Group() as basic_group:
            model_source = gr.Radio(["Pretrained", "Local .pt"], value="Pretrained", label="Model Source")
            pretrained_model = gr.Dropdown(
                label="Pretrained Model (Lazy Download)",
                choices=[],
                value=None,
                visible=True,
            )
            local_model = gr.Textbox(
                label="Local .pt Path",
                placeholder="models/your_model.pt",
                visible=False,
            )
            local_upload = gr.File(label="Upload .pt (saved to models/)", file_types=[".pt"], visible=False)
            upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
            pretrained_hint = gr.Textbox(label="Model Status", interactive=False)
            components.update(
                {
                    "model_source": model_source,
                    "pretrained_model": pretrained_model,
                    "local_model": local_model,
                    "local_upload": local_upload,
                    "upload_status": upload_status,
                    "pretrained_hint": pretrained_hint,
                }
            )

            def _toggle_model_source(kind):
                show_local = kind == "Local .pt"
                return (
                    gr.update(visible=kind == "Pretrained"),
                    gr.update(visible=show_local),
                    gr.update(visible=show_local),
                    gr.update(visible=show_local),
                )

            model_source.change(
                _toggle_model_source,
                inputs=model_source,
                outputs=[pretrained_model, local_model, local_upload, upload_status],
            )

            input_type = gr.Radio(["Images", "Video", "Path", "URL"], value="Images", label="Source Type")
            images = gr.Files(label="Images (multiple)", file_types=["image"], visible=True)
            video = gr.File(label="Video", file_types=["video"], visible=False)
            source_path = gr.Textbox(label="Source Path", placeholder="path/to/images or video.mp4", visible=False)
            source_url = gr.Textbox(label="Source URL", placeholder="https://...", visible=False)
            components.update(
                {
                    "input_type": input_type,
                    "images": images,
                    "video": video,
                    "source_path": source_path,
                    "source_url": source_url,
                }
            )

            def _toggle_source(kind):
                return (
                    gr.update(visible=kind == "Images"),
                    gr.update(visible=kind == "Video"),
                    gr.update(visible=kind == "Path"),
                    gr.update(visible=kind == "URL"),
                )

            input_type.change(
                _toggle_source,
                inputs=input_type,
                outputs=[images, video, source_path, source_url],
            )

            with gr.Row():
                conf = gr.Number(label="Confidence (conf)", value=0.25)
                iou = gr.Number(label="IoU (iou)", value=0.7)
                imgsz = gr.Number(label="Image Size (imgsz)", value=640, precision=0)
            components.update({"conf": conf, "iou": iou, "imgsz": imgsz})

            device_mode_default, gpu_list = device_defaults()
            device_mode = gr.Radio(
                ["auto", "cpu", "single", "multi"],
                value=device_mode_default,
                label="Device Mode",
            )
            single_gpu = gr.Dropdown(
                label="Single GPU",
                choices=gpu_list,
                value=gpu_list[0] if gpu_list else None,
                visible=device_mode_default == "single",
            )
            multi_gpu = gr.CheckboxGroup(
                label="Multi GPU",
                choices=gpu_list,
                value=gpu_list[:1] if gpu_list else [],
                visible=device_mode_default == "multi",
            )
            components.update(
                {
                    "device_mode": device_mode,
                    "single_gpu": single_gpu,
                    "multi_gpu": multi_gpu,
                }
            )

            def _toggle_devices(mode):
                return (
                    gr.update(visible=mode == "single"),
                    gr.update(visible=mode == "multi"),
                )

            device_mode.change(_toggle_devices, inputs=device_mode, outputs=[single_gpu, multi_gpu])

            save = gr.Checkbox(label="Save Outputs", value=True)
            components["save"] = save

            start_btn = gr.Button("Run Predict", elem_classes=["accent-btn"])
            stop_btn = gr.Button("Stop", elem_classes=["secondary-btn"])
            log_box = gr.Textbox(label="Predict Logs", lines=14)
            output_gallery = gr.Gallery(label="Output Images", columns=3)
            output_video = gr.Video(label="Output Video")
            output_dir = gr.Textbox(label="Output Directory", interactive=False)
            components.update(
                {
                    "start_btn": start_btn,
                    "stop_btn": stop_btn,
                    "log_box": log_box,
                    "output_gallery": output_gallery,
                    "output_video": output_video,
                    "output_dir": output_dir,
                }
            )

            cli_preview_basic = gr.Textbox(label="CLI Preview", interactive=False)
            components["cli_preview_basic"] = cli_preview_basic

        with gr.Group(visible=False) as advanced_group:
            adv_model_source = gr.Radio(["Pretrained", "Local .pt"], value="Pretrained", label="Model Source")
            adv_pretrained_model = gr.Dropdown(
                label="Pretrained Model (Lazy Download)",
                choices=[],
                value=None,
                visible=True,
            )
            adv_local_model = gr.Textbox(
                label="Local .pt Path",
                placeholder="models/your_model.pt",
                visible=False,
            )
            adv_local_upload = gr.File(label="Upload .pt (saved to models/)", file_types=[".pt"], visible=False)
            adv_upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
            adv_pretrained_hint = gr.Textbox(label="Model Status", interactive=False)
            adv_input_type = gr.Radio(["Images", "Video", "Path", "URL"], value="Images", label="Source Type")
            adv_images = gr.Files(label="Images (multiple)", file_types=["image"], visible=True)
            adv_video = gr.File(label="Video", file_types=["video"], visible=False)
            adv_source_path = gr.Textbox(label="Source Path", placeholder="path/to/images or video.mp4", visible=False)
            adv_source_url = gr.Textbox(label="Source URL", placeholder="https://...", visible=False)
            components.update(
                {
                    "adv_model_source": adv_model_source,
                    "adv_pretrained_model": adv_pretrained_model,
                    "adv_local_model": adv_local_model,
                    "adv_local_upload": adv_local_upload,
                    "adv_upload_status": adv_upload_status,
                    "adv_pretrained_hint": adv_pretrained_hint,
                    "adv_input_type": adv_input_type,
                    "adv_images": adv_images,
                    "adv_video": adv_video,
                    "adv_source_path": adv_source_path,
                    "adv_source_url": adv_source_url,
                }
            )

            def _toggle_adv_model_source(kind):
                show_local = kind == "Local .pt"
                return (
                    gr.update(visible=kind == "Pretrained"),
                    gr.update(visible=show_local),
                    gr.update(visible=show_local),
                    gr.update(visible=show_local),
                )

            adv_model_source.change(
                _toggle_adv_model_source,
                inputs=adv_model_source,
                outputs=[adv_pretrained_model, adv_local_model, adv_local_upload, adv_upload_status],
            )

            def _toggle_adv_source(kind):
                return (
                    gr.update(visible=kind == "Images"),
                    gr.update(visible=kind == "Video"),
                    gr.update(visible=kind == "Path"),
                    gr.update(visible=kind == "URL"),
                )

            adv_input_type.change(
                _toggle_adv_source,
                inputs=adv_input_type,
                outputs=[adv_images, adv_video, adv_source_path, adv_source_url],
            )

            search = gr.Textbox(label="Search Params (key)", placeholder="conf, iou, vid_stride...")
            components["adv_search"] = search
            adv_flat, adv_grouped = _build_advanced_inputs()
            components["adv_flat"] = adv_flat
            components["adv_grouped"] = adv_grouped

            adv_start = gr.Button("Run Predict (Advanced)", elem_classes=["accent-btn"])
            adv_stop = gr.Button("Stop", elem_classes=["secondary-btn"])
            adv_log = gr.Textbox(label="Predict Logs", lines=14)
            adv_gallery = gr.Gallery(label="Output Images", columns=3)
            adv_video = gr.Video(label="Output Video")
            adv_output_dir = gr.Textbox(label="Output Directory", interactive=False)
            cli_preview_adv = gr.Textbox(label="CLI Preview", interactive=False)
            components.update(
                {
                    "adv_start": adv_start,
                    "adv_stop": adv_stop,
                    "adv_log": adv_log,
                    "adv_gallery": adv_gallery,
                    "adv_video": adv_video,
                    "adv_output_dir": adv_output_dir,
                    "cli_preview_adv": cli_preview_adv,
                }
            )

            def _search_filter(query):
                query = (query or "").strip().lower()
                updates = []
                for key, comp in adv_flat.items():
                    visible = not query or query in key.lower()
                    updates.append(gr.update(visible=visible))
                return updates

            search.change(_search_filter, inputs=search, outputs=list(adv_flat.values()))

        components["basic_group"] = basic_group
        components["advanced_group"] = advanced_group

    return components
