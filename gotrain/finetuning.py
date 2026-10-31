"""Fine-tuning strategy: which parameters of a model actually receive
gradients, shared by both training templates so freezing logic can't
drift between them (same reasoning as gotrain/config_schema.py being the
one place optimizer/scheduler/loss names are validated).

Strategies (see gotrain.config_schema.FINE_TUNING_STRATEGIES):
  - "from_scratch": every parameter is trainable; the caller (the
    template's model-construction code) is responsible for also forcing
    pretrained=False -- this module only ever touches requires_grad, never
    weight initialization.
  - "linear_probe": freeze the entire backbone; only the task-specific
    prediction head is trainable.
  - "partial": freeze everything except the LAST `partial_unfreeze_fraction`
    of the backbone's own parameter list (an approximation of "later
    layers" -- the two model families below expose their internal block
    structure differently, so an exact per-block split isn't attempted)
    plus the head.
  - "full": every parameter is trainable (the behavior every config
    written before this module existed already had).

Freezing is applied directly on the model's own nn.Parameter objects --
no wrapper module, no monkeypatching -- so a forward/backward pass behaves
exactly as if a frozen parameter's gradient were never computed, because
it isn't. trainable_parameters() is the ONLY thing that should ever reach
build_optimizer: handing a frozen parameter to the optimizer wastes memory
on per-parameter state (e.g. Adam's momentum buffers) it will never use.
"""

from typing import Any, Optional

# Head-parameter name prefixes per model family. timm classification
# models (train_classification.py::build_model) name their final layer
# "fc" (resnet family) or "classifier"/"head" (efficientnet/convnext
# family); all of them get reset to num_classes=n_classes at construction
# time regardless of `pretrained`, so these are always the task-specific
# head, never part of the pretrained backbone. segmentation_models_pytorch
# names the task head "segmentation_head." for every
# architecture/encoder combination it supports.
HEAD_PARAM_PREFIXES_BY_FAMILY = {
    "timm": ("fc.", "classifier.", "head."),
    "smp": ("segmentation_head.",),
}

DEFAULT_PARTIAL_UNFREEZE_FRACTION = 0.3


def classify_parameters(model, family: str) -> dict:
    """Split model.named_parameters() into {"head": [...], "backbone": [...]}
    by name prefix. `family` is "timm" (train_classification.py's models)
    or "smp" (train_segmentation.py's).
    """
    if family not in HEAD_PARAM_PREFIXES_BY_FAMILY:
        raise ValueError(f"family must be one of {list(HEAD_PARAM_PREFIXES_BY_FAMILY)}, got {family!r}")
    prefixes = HEAD_PARAM_PREFIXES_BY_FAMILY[family]
    head, backbone = [], []
    for name, param in model.named_parameters():
        (head if name.startswith(prefixes) else backbone).append((name, param))
    return {"head": head, "backbone": backbone}


def apply_fine_tuning_strategy(
    model,
    family: str,
    strategy: str,
    partial_unfreeze_fraction: float = DEFAULT_PARTIAL_UNFREEZE_FRACTION,
) -> dict:
    """Set requires_grad on every parameter of `model` per `strategy` and
    return a summary dict (total/trainable/frozen parameter counts) meant
    to be logged and included in the run's metrics.json.
    """
    groups = classify_parameters(model, family)
    head_params, backbone_params = groups["head"], groups["backbone"]

    if strategy in ("full", "from_scratch"):
        for _, p in head_params + backbone_params:
            p.requires_grad = True
    elif strategy == "linear_probe":
        for _, p in backbone_params:
            p.requires_grad = False
        for _, p in head_params:
            p.requires_grad = True
    elif strategy == "partial":
        n_unfreeze = max(1, int(len(backbone_params) * partial_unfreeze_fraction)) if backbone_params else 0
        frozen_part = backbone_params[: len(backbone_params) - n_unfreeze] if n_unfreeze else backbone_params
        unfrozen_part = backbone_params[len(backbone_params) - n_unfreeze :] if n_unfreeze else []
        for _, p in frozen_part:
            p.requires_grad = False
        for _, p in unfrozen_part:
            p.requires_grad = True
        for _, p in head_params:
            p.requires_grad = True
    else:
        raise ValueError(f"unknown fine_tuning_strategy {strategy!r}")

    all_params = head_params + backbone_params
    total = sum(p.numel() for _, p in all_params)
    trainable = sum(p.numel() for _, p in all_params if p.requires_grad)
    return {
        "fine_tuning_strategy": strategy,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_fraction": round(trainable / total, 4) if total else 0.0,
    }


def trainable_parameters(model):
    """The only parameter list that should ever reach build_optimizer."""
    return [p for p in model.parameters() if p.requires_grad]
