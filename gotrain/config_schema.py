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

TASK_TYPES = ["classification", "segmentation", "fake"]

CLASSIFICATION_VALID_VALUES = {
    "optimizer": ["adam", "sgd"],
    "augmentation": ["none", "basic", "strong"],
    "backbone": ["resnet18"],
}

SEGMENTATION_VALID_VALUES = {
    "architecture": ["unet", "fpn"],
    "optimizer": ["adam", "sgd"],
    "loss": ["ce", "dice", "ce+dice"],
    "augmentation": ["none", "basic", "strong"],
    "encoder": ["resnet34"],
}

VALID_VALUES_BY_TASK_TYPE = {
    "classification": CLASSIFICATION_VALID_VALUES,
    "segmentation": SEGMENTATION_VALID_VALUES,
    "fake": {},
}


class ConfigValidationError(ValueError):
    """Raised when a config value isn't recognized, before any training
    subprocess is spawned. gotrain/tools/training.py raises this instead of
    letting the bad value reach the subprocess, and the agent's hook
    (gotrain/agent.py) recognizes this exception type specifically so a
    rejected launch_training call does not count against --max-runs.
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
