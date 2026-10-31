"""Agent-facing wrapper around gotrain.geoguard.run_geoguard: logs the call
and current status the same way every other tool in this package does.
"""

from gotrain import geoguard, status
from gotrain.logging_utils import log_event


def run_geoguard(dataset: str, task_type: str, max_samples: int = 300) -> dict:
    status.set_action("inspecting data (GeoGuard)", where=dataset)
    log_event("geoguard_started", dataset=dataset, task_type=task_type, max_samples=max_samples)

    full_report = geoguard.run_geoguard(dataset, task_type, max_samples=max_samples)

    log_event(
        "geoguard_finished",
        dataset=dataset,
        task_type=task_type,
        n_findings=len(full_report["findings"]),
        recommended_config=full_report["recommended_config"],
    )
    status.set_last_event(f"GeoGuard inspected {dataset}: {len(full_report['findings'])} finding(s)")
    return full_report
