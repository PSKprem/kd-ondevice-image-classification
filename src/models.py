"""P25 — model builders: ResNet-50 teacher and MobileNetV2 student.

torchvision ships ImageNet-pretrained weights for ResNet-50 and for MobileNetV2
at width 1.0 only. For other widths (0.35 / 0.5 / 0.75) we fall back to timm if
available, otherwise build the width from scratch (randomly initialized).
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
    """Try timm for non-1.0 widths. Returns None if timm/model unavailable."""
    try:
        import timm
    except ImportError:
        return None
    # timm names: mobilenetv2_035, _050, _075, _100, _140 ...
    name = f"mobilenetv2_{int(round(width_mult * 100)):03d}"
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:  # unknown name or no pretrained weights
        warnings.warn(f"timm could not build {name!r} (pretrained={pretrained}): {e}")
        try:
            return timm.create_model(name, pretrained=False, num_classes=num_classes)
        except Exception:
            return None


def build_mobilenetv2(num_classes: int, width_mult: float = 1.0,
                      pretrained: bool = True) -> nn.Module:
    """MobileNetV2 student at the requested width.

    width 1.0 -> torchvision (pretrained available).
    other widths -> timm if installed, else torchvision API from scratch.
    """
    if abs(width_mult - 1.0) < 1e-6:
        return _build_mobilenetv2_torchvision(num_classes, pretrained)

    timm_model = _build_mobilenetv2_timm(num_classes, width_mult, pretrained)
    if timm_model is not None:
        return timm_model

    warnings.warn(
        f"Building MobileNetV2 width={width_mult} from scratch "
        "(no pretrained weights for this width)."
    )
    model = models.MobileNetV2(width_mult=width_mult, num_classes=num_classes)
    return model


def build_model(name: str, num_classes: int, width_mult: float = 1.0,
                pretrained: bool = True) -> nn.Module:
    """Factory used by train/evaluate. name in {resnet50, mobilenetv2}."""
    name = name.lower()
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained)
    if name in ("mobilenetv2", "mbv2"):
        return build_mobilenetv2(num_classes, width_mult, pretrained)
    raise ValueError(f"Unknown model: {name!r}")
