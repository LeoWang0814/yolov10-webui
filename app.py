import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from core.args_schema import coerce_dict, default_cfg_dict
from core.model_zoo import ensure_model, is_model_cached, model_choices
from core.runner import build_command, start_process, stop_process, stream_logs, write_run_metadata
from ui.components import status_bar
from ui.predict import build_predict_tab
from ui.train import build_train_tab


ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = default_cfg_dict()


def _load_css() -> str:
    css_path = ROOT / "ui" / "theme.css"
    return css_path.read_text(encoding="utf-8")


def _refresh_model_choices() -> Tuple[Dict[str, str], Dict[str, Dict]]:
    choices, meta_map = model_choices()
    return choices, meta_map


MODEL_CHOICES, MODEL_META = _refresh_model_choices()


def _model_path_from_choice(choice_label: str) -> Optional[Path]:
    if not choice_label:
        return None
    model_key = MODEL_CHOICES.get(choice_label)
    if not model_key:
        return None
    meta = MODEL_META.get(model_key)
    if not meta:
        return None
    return Path("weights") / meta["release"] / meta["filename"]


def _model_hint(choice_label: Optional[str]) -> str:
    if not choice_label:
        return "Select a pretrained model to see download status."
    model_key = MODEL_CHOICES.get(choice_label)
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
        model_key = MODEL_CHOICES.get(pretrained_label)
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


def _collect_outputs(run_dir: Path) -> Tuple[List[str], Optional[str]]:
    images = []
    video = None
    for suffix in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        images.extend([str(p) for p in run_dir.rglob(suffix)])
    for suffix in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        candidates = list(run_dir.rglob(suffix))
        if candidates:
            video = str(candidates[0])
            break
    return images, video


def _resolve_actual_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        has_files = any(run_dir.rglob("*"))
        if has_files:
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
    if name:
        run_dir = Path("runs") / mode / name
    else:
        run_dir = Path("runs") / mode / "preview"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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


def _extract_results_dir(log_text: str) -> Optional[Path]:
    for line in reversed(log_text.splitlines()):
        if "Results saved to" in line:
            cleaned = line.split("Results saved to", 1)[-1].strip()
            cleaned = cleaned.strip().strip(".")
            if cleaned:
                return Path(cleaned)
    return None


def main() -> gr.Blocks:
    css = _load_css()
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            """
            <div class="app-hero">
              <h1>YOLOv10 WebUI</h1>
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
            if not path:
                return "Missing data path."
            return "OK" if Path(path).exists() else "File not found."

        train_ui["validate_btn"].click(
            _validate_data,
            inputs=train_ui["data_path"],
            outputs=train_ui["data_status"],
        )

        def _refresh_dropdowns():
            global MODEL_CHOICES, MODEL_META
            MODEL_CHOICES, MODEL_META = _refresh_model_choices()
            labels = list(MODEL_CHOICES.keys())
            return (
                gr.update(choices=labels),
                gr.update(choices=labels),
                gr.update(choices=labels),
                gr.update(choices=labels),
            )

        demo.load(
            _refresh_dropdowns,
            inputs=None,
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

        train_ui["model_source"].change(
            _update_train_hint,
            inputs=[train_ui["model_source"], train_ui["pretrained_model"], train_ui["local_model"]],
            outputs=train_ui["pretrained_hint"],
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
                "imgsz": int(imgsz),
                "batch": batch_value,
                "device": device,
                "workers": int(workers),
            }
            run_dir = _run_dir_path("train", run_name, create=create_run_dir)
            args["project"] = str(run_dir.parent)
            args["name"] = run_dir.name
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

        def _run_basic_train(*inputs, progress=gr.Progress()):
            log = ""
            try:
                args = _basic_train_args(*inputs, create_run_dir=True)
                model_path = _resolve_model_path(
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    progress=progress,
                    allow_download=True,
                )
                args["model"] = str(model_path)
                run_dir = Path(args["project"]) / args["name"]
                cmd, preview = build_command("detect", "train", args)
                write_run_metadata(run_dir, args, preview)
                process = start_process(cmd)
                for line in stream_logs(process):
                    clean_line = _strip_ansi(line)
                    log += clean_line
                    yield log, str(run_dir), "", ""
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                yield log, str(run_dir), str(best) if best.exists() else "", str(last) if last.exists() else ""
            except Exception as exc:
                _log_exception("basic train", exc)
                yield f"{log}\nError: {exc}", "", "", ""

        train_ui["start_btn"].click(
            _run_basic_train,
            inputs=basic_train_inputs,
            outputs=[
                train_ui["log_box"],
                train_ui["output_dir"],
                train_ui["best_path"],
                train_ui["last_path"],
            ],
        )

        def _stop_train():
            stop_process()
            return "Stopped by user."

        train_ui["stop_btn"].click(_stop_train, inputs=None, outputs=train_ui["log_box"])

        def _set_predict_model(best_path, last_path):
            path = best_path or last_path
            if not path:
                return gr.update(value=""), gr.update(value="Local .pt")
            return gr.update(value=path), gr.update(value="Local .pt")

        train_ui["set_predict_btn"].click(
            _set_predict_model,
            inputs=[train_ui["best_path"], train_ui["last_path"]],
            outputs=[predict_ui["local_model"], predict_ui["model_source"]],
        )

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
            log = ""
            try:
                values = _gather_adv_values(train_ui["adv_flat"], list(adv_values))
                model_path = _resolve_model_path(
                    adv_model_source,
                    adv_pretrained_model,
                    adv_local_model,
                    progress=progress,
                    allow_download=True,
                )
                values["model"] = str(model_path)
                if adv_data_path:
                    values["data"] = adv_data_path
                run_dir = _run_dir_path("train", run_name, create=True)
                values["project"] = str(run_dir.parent)
                values["name"] = run_dir.name
                cmd, preview = build_command("detect", "train", values)
                write_run_metadata(run_dir, values, preview)
                process = start_process(cmd)
                for line in stream_logs(process):
                    clean_line = _strip_ansi(line)
                    log += clean_line
                    yield log, str(run_dir), "", ""
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                yield log, str(run_dir), str(best) if best.exists() else "", str(last) if last.exists() else ""
            except Exception as exc:
                _log_exception("advanced train", exc)
                yield f"{log}\nError: {exc}", "", "", ""

        train_ui["adv_start"].click(
            _run_adv_train,
            inputs=adv_train_inputs,
            outputs=[
                train_ui["adv_log"],
                train_ui["adv_output_dir"],
                train_ui["adv_best_path"],
                train_ui["adv_last_path"],
            ],
        )
        train_ui["adv_stop"].click(_stop_train, inputs=None, outputs=train_ui["adv_log"])
        train_ui["adv_set_predict_btn"].click(
            _set_predict_model,
            inputs=[train_ui["adv_best_path"], train_ui["adv_last_path"]],
            outputs=[predict_ui["local_model"], predict_ui["model_source"]],
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
            save,
        ) -> Dict:
            model_path = _resolve_model_path(model_source, pretrained_model, local_model, allow_download=False)
            device = _device_value(device_mode, single_gpu, multi_gpu)
            args = {
                "model": str(model_path),
                "conf": conf,
                "iou": iou,
                "imgsz": int(imgsz),
                "device": device,
                "save": bool(save),
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
            predict_ui["save"],
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
                args = _basic_predict_args(*inputs)
                model_path = _resolve_model_path(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    progress=progress,
                    allow_download=True,
                )
                args["model"] = str(model_path)
                run_dir = _run_dir_path("predict", None, create=True)
                source = _prepare_source(inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], run_dir, preview=False)
                args["source"] = source
                args["project"] = str(run_dir.parent)
                args["name"] = run_dir.name
                cmd, preview = build_command("detect", "predict", args)
                write_run_metadata(run_dir, args, preview)
                process = start_process(cmd)
                for line in stream_logs(process):
                    clean_line = _strip_ansi(line)
                    log += clean_line
                    yield log, [], None, str(run_dir)
                parsed_dir = _extract_results_dir(log)
                actual_dir = parsed_dir if parsed_dir else _resolve_actual_run_dir(run_dir)
                images, video = _collect_outputs(actual_dir)
                yield log, images, video, str(actual_dir.resolve())
            except Exception as exc:
                _log_exception("basic predict", exc)
                yield f"{log}\nError: {exc}", [], None, ""

        predict_ui["start_btn"].click(
            _run_basic_predict,
            inputs=basic_predict_inputs,
            outputs=[
                predict_ui["log_box"],
                predict_ui["output_gallery"],
                predict_ui["output_video"],
                predict_ui["output_dir"],
            ],
        )

        predict_ui["stop_btn"].click(_stop_train, inputs=None, outputs=predict_ui["log_box"])

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
                values = _gather_adv_values(predict_ui["adv_flat"], list(adv_values))
                model_path = _resolve_model_path(
                    adv_model_source,
                    adv_pretrained_model,
                    adv_local_model,
                    progress=progress,
                    allow_download=True,
                )
                values["model"] = str(model_path)
                run_dir = _run_dir_path("predict", None, create=True)
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
                process = start_process(cmd)
                for line in stream_logs(process):
                    clean_line = _strip_ansi(line)
                    log += clean_line
                    yield log, [], None, str(run_dir)
                parsed_dir = _extract_results_dir(log)
                actual_dir = parsed_dir if parsed_dir else _resolve_actual_run_dir(run_dir)
                images, video = _collect_outputs(actual_dir)
                yield log, images, video, str(actual_dir.resolve())
            except Exception as exc:
                _log_exception("advanced predict", exc)
                yield f"{log}\nError: {exc}", [], None, ""

        predict_ui["adv_start"].click(
            _run_adv_predict,
            inputs=adv_predict_inputs,
            outputs=[
                predict_ui["adv_log"],
                predict_ui["adv_gallery"],
                predict_ui["adv_video"],
                predict_ui["adv_output_dir"],
            ],
        )
        predict_ui["adv_stop"].click(_stop_train, inputs=None, outputs=predict_ui["adv_log"])

        gr.HTML(
            "<div class='app-footer'>WebUI developed by <a href='https://github.com/LeoWang0814' target='_blank'>LeoWang</a></div>"
        )

    return demo


if __name__ == "__main__":
    main().queue().launch()
