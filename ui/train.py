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
    grouped_defaults = build_grouped_defaults("train")
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


def build_train_tab() -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    with gr.Tab("Train"):
        with gr.Group() as basic_group:
            with gr.Row():
                data_path = gr.Textbox(label="Dataset YAML Path (data)", placeholder="datasets/coco8.yaml")
                validate_btn = gr.Button("Validate Data")
                data_status = gr.HTML(label="Data Status")
            components.update(
                {
                    "data_path": data_path,
                    "validate_btn": validate_btn,
                    "data_status": data_status,
                }
            )

            with gr.Row():
                model_source = gr.Radio(["Pretrained", "Local .pt"], value="Pretrained", label="Model Source")
            with gr.Row():
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
            with gr.Row():
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

            with gr.Row():
                epochs = gr.Number(label="Epochs", value=100, precision=0)
                imgsz = gr.Number(label="Image Size (imgsz)", value=640, precision=0)
                batch = gr.Dropdown(label="Batch", choices=["auto", 1, 2, 4, 8, 16, 32, 64], value=16)
                workers = gr.Number(label="Workers", value=8, precision=0)
            components.update(
                {
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "workers": workers,
                }
            )

            device_mode_default, gpu_list = device_defaults()
            with gr.Row():
                device_mode = gr.Radio(
                    ["auto", "cpu", "single", "multi"],
                    value=device_mode_default,
                    label="Device Mode",
                )
            with gr.Row():
                single_gpu = gr.Dropdown(
                    label="Single GPU",
                    choices=gpu_list,
                    value=gpu_list[0] if gpu_list else None,
                    visible=device_mode_default == "single",
                )
                multi_gpu = gr.CheckboxGroup(
                    label="Multi GPU (DDP)",
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
                show_single = mode == "single"
                show_multi = mode == "multi"
                return (
                    gr.update(visible=show_single),
                    gr.update(visible=show_multi),
                )

            device_mode.change(_toggle_devices, inputs=device_mode, outputs=[single_gpu, multi_gpu])

            with gr.Row():
                run_name = gr.Textbox(label="Run Name (optional)", placeholder="exp-001")
                output_dir = gr.Textbox(label="Output Directory", interactive=False)
            components.update({"run_name": run_name, "output_dir": output_dir})
            with gr.Group(visible=False) as basic_conflict_group:
                basic_conflict_message = gr.Markdown("")
                basic_conflict_action = gr.Radio(
                    ["Rename run", "Cancel"],
                    value="Rename run",
                    label="Directory exists",
                )
                basic_conflict_name = gr.Textbox(
                    label="New Run Name",
                    placeholder="exp-001-new",
                )
                with gr.Row():
                    basic_conflict_confirm = gr.Button("Confirm", elem_classes=["accent-btn"])
                    basic_conflict_cancel = gr.Button("Close", elem_classes=["secondary-btn"])
            components.update(
                {
                    "basic_conflict_group": basic_conflict_group,
                    "basic_conflict_message": basic_conflict_message,
                    "basic_conflict_action": basic_conflict_action,
                    "basic_conflict_name": basic_conflict_name,
                    "basic_conflict_confirm": basic_conflict_confirm,
                    "basic_conflict_cancel": basic_conflict_cancel,
                }
            )

            with gr.Row():
                start_btn = gr.Button("Start Training", elem_classes=["accent-btn"])
                stop_btn = gr.Button("Stop", elem_classes=["secondary-btn"])
            log_box = gr.Textbox(label="Training Logs", lines=18)
            best_path = gr.Textbox(label="best.pt", interactive=False)
            last_path = gr.Textbox(label="last.pt", interactive=False)
            set_predict_btn = gr.Button("Set as Predict Model")
            components.update(
                {
                    "start_btn": start_btn,
                    "stop_btn": stop_btn,
                    "log_box": log_box,
                    "best_path": best_path,
                    "last_path": last_path,
                    "set_predict_btn": set_predict_btn,
                }
            )

            cli_preview_basic = gr.Textbox(label="CLI Preview", interactive=False)
            components["cli_preview_basic"] = cli_preview_basic

        with gr.Group(visible=False) as advanced_group:
            adv_data_path = gr.Textbox(label="Dataset YAML Path (data)", placeholder="datasets/coco8.yaml")
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
            adv_run_name = gr.Textbox(label="Run Name (optional)", placeholder="exp-adv-001")
            components.update(
                {
                    "adv_data_path": adv_data_path,
                    "adv_model_source": adv_model_source,
                    "adv_pretrained_model": adv_pretrained_model,
                    "adv_local_model": adv_local_model,
                    "adv_local_upload": adv_local_upload,
                    "adv_upload_status": adv_upload_status,
                    "adv_pretrained_hint": adv_pretrained_hint,
                    "adv_run_name": adv_run_name,
                }
            )
            with gr.Group(visible=False) as adv_conflict_group:
                adv_conflict_message = gr.Markdown("")
                adv_conflict_action = gr.Radio(
                    ["Rename run", "Cancel"],
                    value="Rename run",
                    label="Directory exists",
                )
                adv_conflict_name = gr.Textbox(
                    label="New Run Name",
                    placeholder="exp-adv-001-new",
                )
                with gr.Row():
                    adv_conflict_confirm = gr.Button("Confirm", elem_classes=["accent-btn"])
                    adv_conflict_cancel = gr.Button("Close", elem_classes=["secondary-btn"])
            components.update(
                {
                    "adv_conflict_group": adv_conflict_group,
                    "adv_conflict_message": adv_conflict_message,
                    "adv_conflict_action": adv_conflict_action,
                    "adv_conflict_name": adv_conflict_name,
                    "adv_conflict_confirm": adv_conflict_confirm,
                    "adv_conflict_cancel": adv_conflict_cancel,
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
            search = gr.Textbox(label="Search Params (key)", placeholder="imgsz, lr0, augment...")
            components["adv_search"] = search
            adv_flat, adv_grouped = _build_advanced_inputs()
            components["adv_flat"] = adv_flat
            components["adv_grouped"] = adv_grouped
            adv_log = gr.Textbox(label="Training Logs", lines=18)
            adv_start = gr.Button("Start Training (Advanced)", elem_classes=["accent-btn"])
            adv_stop = gr.Button("Stop", elem_classes=["secondary-btn"])
            adv_output_dir = gr.Textbox(label="Output Directory", interactive=False)
            adv_best_path = gr.Textbox(label="best.pt", interactive=False)
            adv_last_path = gr.Textbox(label="last.pt", interactive=False)
            adv_set_predict_btn = gr.Button("Set as Predict Model")
            cli_preview_adv = gr.Textbox(label="CLI Preview", interactive=False)
            components.update(
                {
                    "adv_log": adv_log,
                    "adv_start": adv_start,
                    "adv_stop": adv_stop,
                    "adv_output_dir": adv_output_dir,
                    "adv_best_path": adv_best_path,
                    "adv_last_path": adv_last_path,
                    "adv_set_predict_btn": adv_set_predict_btn,
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

        with gr.Group(elem_classes=["metrics-panel"]):
            with gr.Row():
                gr.Markdown("### Logs / Metrics")
                view_range = gr.Number(label="View Range (rows, -1=all)", value=-1, precision=0)
                metrics_refresh = gr.Button(value="Click to refresh charts", elem_id="metrics-refresh")
            metrics_timer = gr.Timer(1.0) if hasattr(gr, "Timer") else None
            with gr.Tabs():
                with gr.Tab("Loss"):
                    loss_plot = gr.Plot()
                with gr.Tab("Metrics"):
                    metric_plot = gr.Plot()
                with gr.Tab("LR"):
                    lr_plot = gr.Plot()
            with gr.Row():
                table_filter = gr.Textbox(label="Search (rows/cols)", placeholder="search rows or cols=loss,map")
                export_btn = gr.Button("Export CSV", elem_classes=["secondary-btn"])
                export_file = gr.File(label="Download CSV", interactive=False)
            metrics_table = gr.Dataframe(label="Logs", interactive=False, wrap=True)
            components.update(
                {
                    "view_range": view_range,
                    "metrics_refresh": metrics_refresh,
                    "metrics_timer": metrics_timer,
                    "loss_plot": loss_plot,
                    "metric_plot": metric_plot,
                    "lr_plot": lr_plot,
                    "table_filter": table_filter,
                    "export_btn": export_btn,
                    "export_file": export_file,
                    "metrics_table": metrics_table,
                }
            )

    return components
