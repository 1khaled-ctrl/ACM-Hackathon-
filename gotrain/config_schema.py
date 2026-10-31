"""Shared registry of valid config values for training runs.

Both gotrain/tools/training.py (pre-flight validation, before any run is
counted against --max-runs) and the training templates themselves
(train_classification.py / train_segmentation.py) validate against this
same registry, so there is exactly one place to update when a new
optimizer/scheduler/loss/backbone option is added — the two can't drift
out of sync with each other or with what the agent is told in the system
prompt.

Every entry here is additive: a config key not present keeps using the
trainer's own default, so old configs (saved runs, hardcoded examples)
keep working after this registry grows.
"""

from typing import Optional

TASK_TYPES = ["classification", "segmentation", "fake"]

# Fine-tuning strategy (see gotrain/finetuning.py): which parameters of a
# pretrained backbone actually get gradients. "full" (everything trainable)
# is the default -- it matches every config written before this field
# existed, so old configs/examples keep behaving exactly as before.
FINE_TUNING_STRATEGIES = ["from_scratch", "linear_probe", "partial", "full"]

CLASSIFICATION_VALID_VALUES = {
    "optimizer": ["adam", "sgd", "adamw"],
    "scheduler": ["none", "cosine"],
    "augmentation": ["none", "basic", "strong"],
    "backbone": ["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"],
    "fine_tuning_strategy": FINE_TUNING_STRATEGIES,
}

SEGMENTATION_VALID_VALUES = {
    "architecture": ["unet", "fpn"],
    "optimizer": ["adam", "sgd", "adamw"],
    "scheduler": ["none", "cosine"],
    "loss": ["ce", "dice", "ce+dice", "focal", "ce+focal"],
    "augmentation": ["none", "basic", "strong"],
    "encoder": ["resnet34", "resnet50", "efficientnet-b3"],
    "fine_tuning_strategy": FINE_TUNING_STRATEGIES,
}

VALID_VALUES_BY_TASK_TYPE = {
    "classification": CLASSIFICATION_VALID_VALUES,
    "segmentation": SEGMENTATION_VALID_VALUES,
    "fake": {},
}


class PreflightValidationError(ValueError):
    """Base class for anything rejected before a training subprocess is
    spawned (bad config value, malformed hypothesis). gotrain/agent.py's
    limit hook recognizes this base class so a rejected launch_training
    call never counts against --max-runs, regardless of which specific
    preflight check rejected it.
    """


class ConfigValidationError(PreflightValidationError):
    """Raised when a config value isn't recognized, before any training
    subprocess is spawned. gotrain/tools/training.py raises this instead of
    letting the bad value reach the subprocess.
    """

    def __init__(self, task_type: str, field: str, value, valid_values: list):
        self.task_type = task_type
        self.field = field
        self.value = value
        self.valid_values = valid_values
        super().__init__(
            f"Invalid value for {field!r}: {value!r}. "
            f"Valid values for {task_type!r}/{field!r} are: {valid_values}."
        )


def validate_config(task_type: str, config: dict) -> None:
    """Raise ConfigValidationError on the first unrecognized enum-like
    value in `config`, checked against VALID_VALUES_BY_TASK_TYPE[task_type].

    A field not present in the schema (e.g. "lr", "epochs", numeric/bool
    knobs with no fixed enum) is not checked here — this only guards the
    closed-choice fields (optimizer, scheduler, loss, backbone/encoder,
    architecture, augmentation).
    """
    if task_type not in VALID_VALUES_BY_TASK_TYPE:
        raise ConfigValidationError(task_type, "task_type", task_type, TASK_TYPES)

    valid_values = VALID_VALUES_BY_TASK_TYPE[task_type]
    for field, allowed in valid_values.items():
        if field in config and config[field] is not None and config[field] not in allowed:
            raise ConfigValidationError(task_type, field, config[field], allowed)


def validate_fine_tuning_strategy(task_type: str, config: dict) -> None:
    """linear_probe/partial only make sense with a pretrained backbone --
    freezing most or all of a randomly-initialized backbone produces a
    model that cannot learn useful features no matter how long the head
    trains. Rejected here, before any subprocess runs, rather than
    discovered after a run's GPU budget was spent on a config that was
    never going to work.
    """
    strategy = config.get("fine_tuning_strategy") or "full"
    pretrained = config.get("pretrained", True)
    if strategy in ("linear_probe", "partial") and not pretrained:
        raise ConfigValidationError(
            task_type, "fine_tuning_strategy", strategy,
            [f"{strategy!r} requires pretrained=true -- freezing a randomly-initialized "
             "backbone cannot learn useful features"],
        )


def validate_class_weights(task_type: str, class_weights: Optional[list], n_classes: Optional[int]) -> None:
    """`class_weights` has no enum to check against validate_config (it's a
    free-form list[float]), so its one real invariant -- length must equal
    the dataset's actual class count -- was never checked anywhere before
    this. A length mismatch previously reached
    torch.nn.CrossEntropyLoss(weight=...) uncaught (build_loss isn't
    wrapped in either template's OOM try/except), surfacing as a bare,
    confusing RuntimeError deep in a training subprocess that had already
    spent run budget getting there. Rejected here instead, before any
    subprocess is spawned.

    `n_classes=None` (the dataset's class count genuinely couldn't be
    resolved pre-flight) is allowed through un-checked -- there's nothing
    to validate against, and the template's own defense-in-depth check
    will still catch a real mismatch with a clear message instead of a
    cryptic one.
    """
    if class_weights is None or n_classes is None:
        return
    if len(class_weights) != n_classes:
        raise ConfigValidationError(
            task_type, "class_weights", class_weights,
            [f"must have exactly {n_classes} entries (one per class) for this dataset, "
             f"got {len(class_weights)}"],
        )


# ---------------------------------------------------------------------------
# Structured experiment hypotheses (see the system prompt's workflow step 3
# and gotrain/ledger.py's hypothesis_outcome computation). Every
# launch_training call that has a parent_id (i.e. every run after the first
# baseline) must state a hypothesis -- what it changed, why, and what effect
# it expects -- enforced here in code, not left to the prompt, exactly like
# every other guardrail in this module.
# ---------------------------------------------------------------------------

EXPECTED_DIRECTIONS = ["increase", "decrease", "no_change"]
HYPOTHESIS_MAGNITUDES = ["small", "moderate", "large"]
HYPOTHESIS_REQUIRED_TEXT_FIELDS = ["change", "why"]


class HypothesisValidationError(PreflightValidationError):
    """Raised when a launch_training call's hypothesis is missing (while a
    parent_id is given) or malformed (missing/empty required text fields,
    or an expected_direction/expected_magnitude outside the allowed set).
    """


def validate_hypothesis(hypothesis: Optional[dict], has_parent: bool) -> None:
    """Raise HypothesisValidationError on a missing or malformed hypothesis.

    `hypothesis=None` is only allowed when `has_parent` is False -- the
    very first baseline run on a dataset has no prior run to form a
    hypothesis against. Every subsequent run must supply one.
    """
    if hypothesis is None:
        if has_parent:
            raise HypothesisValidationError(
                "hypothesis is required for every run after the first baseline "
                "(parent_id was given, but hypothesis was omitted). State what "
                "you changed vs. the parent run and why you expect it to help."
            )
        return

    if not isinstance(hypothesis, dict):
        raise HypothesisValidationError(
            f"hypothesis must be a dict or null, got {type(hypothesis).__name__}"
        )

    for field in HYPOTHESIS_REQUIRED_TEXT_FIELDS:
        value = hypothesis.get(field)
        if not isinstance(value, str) or not value.strip():
            raise HypothesisValidationError(
                f"hypothesis[{field!r}] must be a non-empty string, got {value!r}"
            )

    direction = hypothesis.get("expected_direction")
    if direction is not None and direction not in EXPECTED_DIRECTIONS:
        raise HypothesisValidationError(
            f"hypothesis['expected_direction'] must be one of {EXPECTED_DIRECTIONS} "
            f"or omitted, got {direction!r}"
        )

    magnitude = hypothesis.get("expected_magnitude")
    if magnitude is not None and magnitude not in HYPOTHESIS_MAGNITUDES:
        raise HypothesisValidationError(
            f"hypothesis['expected_magnitude'] must be one of {HYPOTHESIS_MAGNITUDES} "
            f"or omitted, got {magnitude!r}"
        )
