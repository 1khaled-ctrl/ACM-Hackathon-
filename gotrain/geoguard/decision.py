"""GeoGuard decision engine: turns diagnostics into a preprocessing/config
recommendation drawn ONLY from gotrain.config_schema's allowlisted values.

This deliberately does not (and must not) execute arbitrary image-processing
code decided by an LLM. It is a plain deterministic function: diagnostics
in, a config dict + human-readable findings/rationale out. Every config it
proposes is validated against gotrain.config_schema before being returned,
so a GeoGuard recommendation can never itself be the thing that causes a
launch_training rejection.
"""

from gotrain.config_schema import validate_config

SMALL_DATASET_THRESHOLD = 1000

# Class-imbalance intelligence (see recommend()'s rare_classes branch
# below). CLASS_WEIGHT_CAP bounds how large any single class's weight
# can get -- "do not apply extreme inverse-frequency weights without
# safeguards." Beyond EXTREME_IMBALANCE_RATIO, median-frequency balancing
# (weighting relative to the dataset's typical class frequency) is
# recommended instead of plain inverse-frequency (weighting relative to
# a fixed baseline of 1.0), since it degrades more gracefully when one
# class is vanishingly rare.
CLASS_WEIGHT_CAP = 10.0
EXTREME_IMBALANCE_RATIO = 50.0


def recommend(
    dataset: str,
    task_type: str,
    pixel_diagnostics: dict,
    class_diagnostics: dict,
    band_classification: dict,
    dataset_notes: dict,
) -> dict:
    """Return {"dataset", "task_type", "findings", "rationale",
    "recommended_config"}. `findings` are observations; `rationale`
    explains *why* each recommended_config key was chosen, so a human (or
    the agent) can audit the reasoning, not just the output.
    """
    config: dict = {}
    findings: list = []
    rationale: list = []

    for warning in dataset_notes.get("warnings", []):
        findings.append(warning)

    if band_classification["is_multimodal"]:
        n_sar = len(band_classification["sar_bands"])
        n_optical = len(band_classification["optical_bands"])
        findings.append(
            f"{n_sar} SAR band(s) + {n_optical} optical band(s) detected -- mixed-modality input."
        )
        rationale.append(
            "No band is dropped, reordered, or forced onto a shared scale: the "
            "training template's per-band z-score normalization fits an "
            "independent mean/std for every one of the "
            f"{n_sar + n_optical} channels, so SAR and optical channels are "
            "each normalized against their own statistics even though they "
            "share one normalization *method*. GeoGuard does not recommend a "
            "log/dB transform for the SAR channels because GEO-Bench's stored "
            "representation for this dataset has not been independently "
            "re-derived here -- that would require pixel-level ground truth "
            "this inspection pass doesn't have; flagged for manual sensor-doc "
            "confirmation rather than guessed."
        )

    saturated = [b["band"] for b in pixel_diagnostics["per_band"] if b.get("possible_saturation")]
    if saturated:
        findings.append(f"Possible sensor saturation/clipping in bands: {saturated}")
        rationale.append(
            "Saturated bands are still z-scored like any other band -- GeoGuard "
            "does not auto-clip them, since clipping changes what the model "
            "sees without a human confirming it matches the sensor's known "
            "dynamic range."
        )

    near_constant = [b["band"] for b in pixel_diagnostics["per_band"] if b.get("near_constant")]
    if near_constant:
        findings.append(f"Near-constant band(s) (std < 1e-6): {near_constant}")
        rationale.append(
            "Near-constant bands carry little signal; the training template's "
            "own std-floor guard (std < 1e-6 -> 1.0) already prevents a "
            "divide-by-zero, so no config change is needed, but a near-zero "
            "gradient contribution from these bands is expected."
        )

    if pixel_diagnostics["nan_count"] or pixel_diagnostics["inf_count"]:
        findings.append(
            f"{pixel_diagnostics['nan_count']} NaN and {pixel_diagnostics['inf_count']} Inf "
            "pixel value(s) found in the TRAIN sample."
        )
        rationale.append(
            "NaN/Inf pixels were excluded from the band statistics above so they "
            "don't poison the mean/std, but the training templates do not "
            "currently filter them out of the actual training tensors -- run "
            "geoguard.qa before training on this dataset to confirm they don't "
            "reach the model."
        )

    if class_diagnostics["rare_classes"]:
        findings.append(
            f"Rare classes (<2% of {class_diagnostics['unit']}): {class_diagnostics['rare_classes']}"
        )
        imbalance_ratio = class_diagnostics.get("imbalance_ratio")
        if imbalance_ratio is not None and imbalance_ratio > EXTREME_IMBALANCE_RATIO:
            config["class_weights"] = _median_frequency_weights(class_diagnostics["frequencies"])
            findings.append(
                f"Imbalance ratio is extreme ({imbalance_ratio:.0f}x between the most- and "
                "least-frequent class with any samples)."
            )
            rationale.append(
                f"class_weights set via median-frequency balancing (capped at {CLASS_WEIGHT_CAP:.0f}x, "
                "normalized to sum to n_classes) rather than plain inverse-frequency -- with imbalance "
                "this extreme, inverse-frequency would assign the rarest class a weight wildly "
                "disproportionate to everything else in a batch; median-frequency balancing weights "
                "relative to the dataset's typical class frequency instead. This is a starting point "
                "for a controlled comparison against the unweighted baseline (same architecture/seed, "
                "only class_weights differing), not a guaranteed improvement -- validate both before "
                "keeping either, and track minority-class metrics, not just overall accuracy."
            )
        else:
            config["class_weights"] = _inverse_frequency_weights(class_diagnostics["frequencies"])
            rationale.append(
                f"class_weights set to inverse-frequency (capped at {CLASS_WEIGHT_CAP:.0f}x, normalized "
                "to sum to n_classes) to counter the rare classes found above -- capped so no single "
                "rare class can dominate the loss. A starting point for a controlled comparison against "
                "the unweighted baseline, not a guaranteed improvement."
            )

    if class_diagnostics["background_dominant"]:
        findings.append(
            "Binary segmentation task is background-dominated "
            f"(background is {class_diagnostics['frequencies'][0] * 100:.1f}% of pixels)."
        )
        if task_type == "segmentation":
            config["loss"] = "ce+dice"
            rationale.append(
                "loss=ce+dice recommended over plain ce for a background-dominated "
                "binary mask -- Dice directly optimizes foreground overlap instead "
                "of per-pixel accuracy, which plain CE can trivially minimize by "
                "predicting all-background."
            )

    if pixel_diagnostics["n_samples"] < SMALL_DATASET_THRESHOLD:
        findings.append(
            f"Small TRAIN sample ({pixel_diagnostics['n_samples']} images inspected) -- "
            "elevated overfitting risk."
        )
        config["augmentation"] = "strong"
        rationale.append(
            "augmentation=strong recommended to increase effective training "
            "diversity given the small sample size."
        )

    if dataset == "m-eurosat":
        config["bands"] = None
        rationale.append(
            "bands explicitly left as null (all bands) -- see the band-name "
            "warning above."
        )

    if config:
        validate_config(task_type, config)

    return {
        "dataset": dataset,
        "task_type": task_type,
        "findings": findings,
        "rationale": rationale,
        "recommended_config": config,
    }


def _inverse_frequency_weights(frequencies: list, cap: float = CLASS_WEIGHT_CAP) -> list:
    """weight_c = 1 / freq_c, capped at `cap` before normalizing so no
    single rare class can dominate the loss ("do not apply extreme
    inverse-frequency weights without safeguards").
    """
    raw = [1.0 / f if f > 0 else 0.0 for f in frequencies]
    return _normalize_weights([min(w, cap) for w in raw])


def _median_frequency_weights(frequencies: list, cap: float = CLASS_WEIGHT_CAP) -> list:
    """Median-frequency balancing (Eigen & Fergus): weight_c =
    median(nonzero frequencies) / freq_c. Weights relative to the
    dataset's typical class frequency instead of a fixed baseline of
    1.0, so it degrades more gracefully than plain inverse-frequency
    when one class is vanishingly rare. Also capped at `cap`.
    """
    positive = sorted(f for f in frequencies if f > 0)
    if not positive:
        return [1.0] * len(frequencies)
    median_freq = positive[len(positive) // 2]
    raw = [median_freq / f if f > 0 else 0.0 for f in frequencies]
    return _normalize_weights([min(w, cap) for w in raw])


def _normalize_weights(weights: list) -> list:
    """Normalize so weights sum to n_classes (keeps the loss's overall
    magnitude comparable to unweighted CE instead of scaling it up/down).
    """
    total = sum(weights)
    n = len(weights)
    if total == 0:
        return [1.0] * n
    return [round(w / total * n, 4) for w in weights]
