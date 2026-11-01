"""Proactive failure/instability detection -- surfaces training failures,
timeouts, severe miscalibration, and failed rollbacks from data this
project already logs (events.jsonl, control.json, DataIQ loss-scan
summaries), rather than requiring an engineer to notice them by scrolling
through raw logs.

"Proactive" here means "computed without being asked about a specific
run" -- this project has no email/Slack/push integration to actually
deliver a notification, so check_alerts is the detector an engineer-facing
surface (the dashboard, or a GoAudit report) polls and renders; it is not
itself a delivery channel. gotrain.goaudit's reports include this
module's output in their own "failures" section for exactly that reason.

Pure Python/JSON reads (events.jsonl, control.json) -- no torch needed,
fully testable in this environment.
"""

from typing import Optional

from gotrain import control
from gotrain.logging_utils import read_events

CRITICAL = "critical"
WARNING = "warning"

# 3+ failed runs for the same dataset is treated as a pattern (something
# structurally wrong -- a bad config default, a broken dependency) rather
# than one-off bad luck.
REPEATED_FAILURE_THRESHOLD = 3


def _failed_safe_alert() -> Optional[dict]:
    state = control.get_control()
    if state.get("state") != control.FAILED_SAFE:
        return None
    return {
        "severity": CRITICAL,
        "type": "failed_safe",
        "message": f"System is in FAILED_SAFE: {state.get('reason') or 'no reason recorded'}. "
                   "No training can proceed until an engineer acknowledges this from the dashboard.",
    }


def _training_failure_alerts(dataset: str) -> list:
    alerts = []
    failures = read_events(event="training_failed", dataset=dataset)
    if failures:
        severity = CRITICAL if len(failures) >= REPEATED_FAILURE_THRESHOLD else WARNING
        alerts.append({
            "severity": severity,
            "type": "training_failures",
            "message": f"{len(failures)} training run(s) have failed for {dataset!r} "
                       f"(most recent: {failures[-1]['timestamp']}).",
        })

    timeouts = read_events(event="training_timed_out", dataset=dataset)
    if timeouts:
        alerts.append({
            "severity": WARNING,
            "type": "training_timeouts",
            "message": f"{len(timeouts)} training run(s) exceeded their wall-clock budget "
                       f"for {dataset!r} (most recent: {timeouts[-1]['timestamp']}).",
        })
    return alerts


def _calibration_alerts(dataset: str) -> list:
    alerts = []
    for event in read_events(event="dataiq_loss_scan_finished", dataset=dataset):
        calibration = event.get("calibration")
        verdict = calibration.get("verdict") if calibration else None
        if isinstance(verdict, str) and verdict.startswith("severely"):
            alerts.append({
                "severity": WARNING,
                "type": "severe_miscalibration",
                "run_id": event.get("run_id"),
                "message": f"Run {event.get('run_id')} for {dataset!r} is {verdict} "
                           f"(ECE={calibration.get('ece'):.3f}) -- its confidence scores should not "
                           "be trusted at face value.",
            })
    return alerts


def _rollback_failure_alerts(dataset: str) -> list:
    alerts = []
    for event in read_events(event="rollback_failed", dataset=dataset):
        alerts.append({
            "severity": CRITICAL,
            "type": "rollback_verification_failed",
            "message": f"Rollback to version {event.get('version')} for {dataset!r} failed "
                       f"{event.get('stage')} verification: {event.get('problems')}.",
        })
    return alerts


def check_alerts(dataset: str) -> list:
    """All currently-active alerts for `dataset`, most severe first.
    Empty list means nothing to flag -- not "nothing has ever happened,"
    just nothing matching a known alert condition is present right now.
    """
    alerts = []
    failed_safe = _failed_safe_alert()
    if failed_safe is not None:
        alerts.append(failed_safe)
    alerts.extend(_training_failure_alerts(dataset))
    alerts.extend(_calibration_alerts(dataset))
    alerts.extend(_rollback_failure_alerts(dataset))

    severity_order = {CRITICAL: 0, WARNING: 1}
    return sorted(alerts, key=lambda a: severity_order.get(a["severity"], 99))
