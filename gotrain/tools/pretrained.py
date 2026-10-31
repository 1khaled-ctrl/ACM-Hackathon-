"""Agent-facing wrappers around gotrain.pretrained (the curated
pretrained-model registry). See gotrain/tools/geoguard.py for the
identical logging/status pattern.
"""

from typing import Optional

from gotrain import pretrained, status
from gotrain.logging_utils import log_event


def list_compatible_models(task_type: str, dataset: Optional[str] = None) -> list:
    return pretrained.list_compatible_models(task_type, dataset=dataset)


def estimate_model_resources(model_id: str) -> dict:
    return pretrained.estimate_model_resources(model_id)


def validate_model_compatibility(model_id: str, dataset: str, task_type: str, n_bands: int, n_classes: int) -> dict:
    status.set_action(f"validating compatibility of {model_id} (Model Intelligence)", where=dataset)
    log_event("model_compatibility_check_started", model_id=model_id, dataset=dataset, task_type=task_type)

    result = pretrained.validate_model_compatibility(model_id, dataset, task_type, n_bands, n_classes)

    log_event("model_compatibility_check_finished", model_id=model_id, dataset=dataset, compatible=result["compatible"])
    return result
