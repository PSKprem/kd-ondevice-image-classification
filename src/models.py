"""P25 — model builders: ResNet-50 teacher and MobileNetV2 student.

torchvision ships ImageNet-pretrained weights for ResNet-50, MobileNetV2 (width
1.0 only) and MobileNetV3-Small. timm adds pretrained MobileNetV2 widths 0.5 and
1.4. Widths 0.35 / 0.75 have NO pretrained weights anywhere: they can only be
built with pretrained=False (from-scratch experiment).
"""
from __future__ import annotations

import warnings

import torch.nn as nn
from torchvision import models


def _replace_classifier_linear(module: nn.Linear, num_classes: int) -> nn.Linear:
    return nn.Linear(module.in_features, num_classes)


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-50 teacher with a fresh classification head."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = _replace_classifier_linear(model.fc, num_classes)
    return model


def _build_mobilenetv2_torchvision(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


def _build_mobilenetv2_timm(num_classes: int, width_mult: float, pretrained: bool):
    """Build a non-1.0 width via timm.

    pretrained=True  -> must load ImageNet weights, otherwise raise (never fall
                        back to random init silently; that broke the w0.5 run).
    pretrained=False -> random init. Returns None if timm is not installed.
    timm has pretrained weights only for widths 0.5 / 1.0 / 1.4.
    """
    try:
        import timm
    except ImportError:
        if pretrained:
            raise RuntimeError("timm is required for pretrained MobileNetV2 "
                               f"width={width_mult}. pip install timm")
        return None
    # timm names: mobilenetv2_035, _050, _075, _100, _140 ...
    name = f"mobilenetv2_{int(round(width_mult * 100)):03d}"
    if not pretrained:
        try:
            return timm.create_model(name, pretrained=False, num_classes=num_classes)
        except Exception:
            return None
    try:
        return timm.create_model(name, pretrained=True, num_classes=num_classes)
    except Exception as e:
        raise RuntimeError(
            f"Could not load pretrained weights for {name!r}: {e}\n"
            "Pretrained MobileNetV2 widths are 0.5, 1.0, 1.4. "
            "Use one of those, or set pretrained: false for a from-scratch run."
        ) from e


def build_mobilenetv2(num_classes: int, width_mult: float = 1.0,
                      pretrained: bool = True) -> nn.Module:
    """MobileNetV2 student at the requested width.

    width 1.0 -> torchvision (pretrained available).
    other widths -> timm (pretrained for 0.5 / 1.4), else torchvision from scratch.
    """
    if abs(width_mult - 1.0) < 1e-6:
        return _build_mobilenetv2_torchvision(num_classes, pretrained)

    timm_model = _build_mobilenetv2_timm(num_classes, width_mult, pretrained)
    if timm_model is not None:
        return timm_model

    warnings.warn(f"Building MobileNetV2 width={width_mult} from scratch (pretrained=False).")
    return models.MobileNetV2(width_mult=width_mult, num_classes=num_classes)


def build_mobilenetv3_small(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small (~1M params) with ImageNet weights from torchvision."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


def build_model(name: str, num_classes: int, width_mult: float = 1.0,
                pretrained: bool = True) -> nn.Module:
    """Factory used by train/evaluate.

    name in {resnet50, mobilenetv2, mobilenetv3_small}.
    """
    name = name.lower()
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained)
    if name in ("mobilenetv2", "mbv2"):
        return build_mobilenetv2(num_classes, width_mult, pretrained)
    if name in ("mobilenetv3_small", "mbv3s"):
        return build_mobilenetv3_small(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name!r}")
