from typing import Any, Dict, List

from ultralytics.cfg import DEFAULT_CFG_DICT, smart_value


TRAIN_GROUPS = {
    "Core": [
        "task",
        "mode",
        "data",
        "model",
        "epochs",
        "time",
        "patience",
        "batch",
        "imgsz",
        "device",
        "workers",
        "project",
        "name",
        "exist_ok",
        "pretrained",
        "seed",
        "deterministic",
        "single_cls",
        "rect",
        "resume",
        "amp",
        "fraction",
        "cache",
        "save",
        "save_period",
        "val",
        "val_period",
        "verbose",
    ],
    "Optimizer": [
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "nbs",
        "cos_lr",
    ],
    "Augmentation": [
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "copy_paste",
        "auto_augment",
        "erasing",
        "crop_fraction",
        "close_mosaic",
        "multi_scale",
        "overlap_mask",
        "mask_ratio",
        "dropout",
    ],
    "Validation / Logging": [
        "plots",
        "save_json",
        "save_hybrid",
        "conf",
        "iou",
        "max_det",
        "half",
        "dnn",
        "save_txt",
        "save_conf",
        "save_crop",
        "show_labels",
        "show_conf",
        "show_boxes",
        "line_width",
    ],
    "Performance": [
        "profile",
        "device",
        "workers",
        "batch",
        "imgsz",
        "half",
        "amp",
    ],
}

PREDICT_GROUPS = {
    "Core": [
        "task",
        "mode",
        "model",
        "source",
        "imgsz",
        "conf",
        "iou",
        "device",
        "half",
        "max_det",
        "batch",
    ],
    "Output": [
        "save",
        "save_txt",
        "save_conf",
        "save_crop",
        "show",
        "show_labels",
        "show_conf",
        "show_boxes",
        "line_width",
    ],
    "Performance": [
        "vid_stride",
        "stream_buffer",
        "visualize",
        "augment",
        "agnostic_nms",
        "classes",
        "retina_masks",
    ],
}


def default_cfg_dict() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG_DICT)
    cfg["amp"] = False
    return cfg


def build_grouped_defaults(mode: str) -> Dict[str, Dict[str, Any]]:
    cfg = default_cfg_dict()
    groups = TRAIN_GROUPS if mode == "train" else PREDICT_GROUPS
    grouped: Dict[str, Dict[str, Any]] = {}
    used = set()
    for group_name, keys in groups.items():
        group_items = {}
        for key in keys:
            if key in cfg:
                group_items[key] = cfg[key]
                used.add(key)
        grouped[group_name] = group_items
    other = {k: v for k, v in cfg.items() if k not in used}
    if other:
        grouped["Other"] = other
    return grouped


def coerce_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return bool(smart_value(value))
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(value, str):
        return smart_value(value)
    return value


def coerce_dict(values: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    coerced = {}
    for key, value in values.items():
        default = defaults.get(key)
        coerced[key] = coerce_value(value, default)
    return coerced
