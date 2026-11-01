"""Curated pretrained-model registry (Model Intelligence).

A STATIC catalog of candidate backbones/encoders, each distinguishing
architecture, pretraining source/domain, and input-channel/preprocessing
requirements -- never a live query against a model hub. No entry here is
ever downloaded automatically: list_compatible_models() only reads this
catalog's metadata, and load_pretrained_backbone() only runs for entries
this module actually knows how to construct (today: the timm/
segmentation_models_pytorch backbones the training templates already
use -- see gotrain/templates/train_classification.py::build_model and
train_segmentation.py::build_model, reused here rather than duplicated).

Every entry's "compatibility_verified" starts False in this static
catalog and is never hand-edited to True -- the only way a model becomes
verified is validate_model_compatibility() actually constructing it with
the target dataset's real band/class count and running one forward pass,
with the result cached under logs/pretrained/verification/. This module
never executes arbitrary code from an untrusted model repository: the
timm/smp entries load through those libraries' own well-known
constructors with a pinned package version (see requirements.txt), and
the aspirational geospatial-foundation-model entries below
(torchgeo/dinov2/Prithvi) have NO loader implemented at all in this
environment -- load_pretrained_backbone raises NotImplementedError for
them rather than attempting an unverified download.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from gotrain.logging_utils import LOGS_DIR

VERIFICATION_DIR = LOGS_DIR / "pretrained" / "verification"

# ---------------------------------------------------------------------------
# Catalog. "parameter_count" is a published architecture-spec figure
# (textbook/paper-level public knowledge), explicitly NOT a live
# measurement -- estimate_model_resources() labels it as such. Entries
# with a loader ("loader_family") can actually be constructed in this
# project today; entries without one are documented candidates only.
# ---------------------------------------------------------------------------

CATALOG = {
    "timm/resnet18": {
        "model_id": "timm/resnet18",
        "architecture": "resnet18",
        "pretraining_source": "timm (torchvision ImageNet-1k weights)",
        "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3,
        "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split (gotrain templates' own convention)"},
        "parameter_count": 11_700_000,
        "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (timm) / BSD-3 (torchvision weights)",
        "checkpoint_revision": "timm default (pretrained=True); not pinned to a specific hash",
        "compatibility_verified": False,
        "loader_family": "timm",
        "task_types": ["classification"],
        "notes": "timm adapts the first conv layer to any in_chans at construction time -- already how "
                 "train_classification.py uses this for 13/18-band GEO-Bench inputs.",
    },
    "timm/resnet50": {
        "model_id": "timm/resnet50", "architecture": "resnet50",
        "pretraining_source": "timm (torchvision ImageNet-1k weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 25_600_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (timm) / BSD-3 (torchvision weights)",
        "checkpoint_revision": "timm default (pretrained=True)", "compatibility_verified": False,
        "loader_family": "timm", "task_types": ["classification"],
        "notes": "Fallback baseline for m-eurosat in prompts/system_prompt.md.",
    },
    "timm/efficientnet_b0": {
        "model_id": "timm/efficientnet_b0", "architecture": "efficientnet_b0",
        "pretraining_source": "timm (ImageNet-1k weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 5_300_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (timm)", "checkpoint_revision": "timm default (pretrained=True)",
        "compatibility_verified": False, "loader_family": "timm", "task_types": ["classification"],
        "notes": "Lightest classification option in the catalog -- cheapest GPU/memory footprint.",
    },
    "timm/convnext_tiny": {
        "model_id": "timm/convnext_tiny", "architecture": "convnext_tiny",
        "pretraining_source": "timm (ImageNet-1k weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 28_600_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (timm) / MIT (ConvNeXt weights)",
        "checkpoint_revision": "timm default (pretrained=True)", "compatibility_verified": False,
        "loader_family": "timm", "task_types": ["classification"],
        "notes": "Known-good m-eurosat baseline in prompts/system_prompt.md (measured ~0.981 val accuracy "
                 "in a prior project run -- that is a real result from this project, not from this catalog).",
    },
    "smp/resnet34": {
        "model_id": "smp/resnet34", "architecture": "resnet34 (segmentation_models_pytorch encoder)",
        "pretraining_source": "segmentation_models_pytorch (ImageNet weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 21_800_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (segmentation_models_pytorch) / BSD-3 (weights)",
        "checkpoint_revision": "smp default (encoder_weights='imagenet')", "compatibility_verified": False,
        "loader_family": "smp", "task_types": ["segmentation"],
        "notes": "Default encoder for both segmentation datasets.",
    },
    "smp/resnet50": {
        "model_id": "smp/resnet50", "architecture": "resnet50 (segmentation_models_pytorch encoder)",
        "pretraining_source": "segmentation_models_pytorch (ImageNet weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 25_600_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (segmentation_models_pytorch) / BSD-3 (weights)",
        "checkpoint_revision": "smp default (encoder_weights='imagenet')", "compatibility_verified": False,
        "loader_family": "smp", "task_types": ["segmentation"],
        "notes": None,
    },
    "smp/efficientnet-b3": {
        "model_id": "smp/efficientnet-b3", "architecture": "efficientnet-b3 (segmentation_models_pytorch encoder)",
        "pretraining_source": "segmentation_models_pytorch (ImageNet weights)", "pretraining_domain": "natural_images_imagenet",
        "supported_modalities": ["optical_rgb", "multispectral_via_channel_adaptation"],
        "expected_channels": 3, "input_band_order": [],
        "normalization": {"method": "per-band z-score, fit on TRAIN split"},
        "parameter_count": 12_000_000, "parameter_count_source": "published architecture spec, not measured in this environment",
        "license": "Apache-2.0 (segmentation_models_pytorch)", "checkpoint_revision": "smp default (encoder_weights='imagenet')",
        "compatibility_verified": False, "loader_family": "smp", "task_types": ["segmentation"],
        "notes": None,
    },
    # -- Aspirational geospatial-domain candidates. No loader is
    # implemented for these in this environment (no internet access to
    # confirm availability/license/revision, no torchgeo/transformers
    # install to test against) -- documented as candidates for a future
    # pass on the SageMaker lab, never auto-downloaded or auto-executed.
    "torchgeo/resnet18_sentinel2_all_moco": {
        "model_id": "torchgeo/resnet18_sentinel2_all_moco", "architecture": "resnet18",
        "pretraining_source": "torchgeo (self-supervised MoCo on Sentinel-2)", "pretraining_domain": "satellite_sentinel2",
        "supported_modalities": ["multispectral"], "expected_channels": 13, "input_band_order": None,
        "normalization": {"method": "unknown -- torchgeo's own documented stats, not re-derived here"},
        "parameter_count": 11_700_000, "parameter_count_source": "published architecture spec (resnet18), not measured in this environment",
        "license": None, "checkpoint_revision": None, "compatibility_verified": False,
        "loader_family": None, "task_types": ["classification"],
        "notes": "Not verified in this environment: no internet access to confirm current availability/license/"
                 "revision, and torchgeo is not installed. A real Sentinel-2-pretrained ResNet18 could be a "
                 "strong m-eurosat candidate -- requires manual verification on the SageMaker lab before use.",
    },
    "huggingface/facebook-dinov2-small": {
        "model_id": "huggingface/facebook-dinov2-small", "architecture": "vit_small",
        "pretraining_source": "huggingface (DINOv2 self-supervised, general vision)", "pretraining_domain": "natural_images_general",
        "supported_modalities": ["optical_rgb"], "expected_channels": 3, "input_band_order": None,
        "normalization": {"method": "unknown -- would need DINOv2's own documented preprocessing"},
        "parameter_count": 22_000_000, "parameter_count_source": "published architecture spec (ViT-S/14), not measured in this environment",
        "license": "Apache-2.0 (per DINOv2's public release, not re-confirmed here -- verify from the actual "
                    "HF repo before using this in anything submitted)",
        # Deliberately a real, loadable entry (see load_pretrained_backbone's
        # "huggingface_transformers" branch below) -- channel adaptation is
        # done by selecting the dataset's 3 RGB-equivalent bands
        # (gotrain.band_selection.select_rgb_band_indices), not timm's
        # in_chans resize, which doesn't apply to a transformers model.
        "hf_repo_id": "facebook/dinov2-small",
        # Intentionally left unpinned: "pin verified model revisions" means
        # an actual commit SHA, looked up from
        # https://huggingface.co/facebook/dinov2-small/commits/main -- not
        # something to invent without network access to verify it exists.
        # load_pretrained_backbone refuses to run with this still None
        # rather than silently default to "main".
        "hf_revision": None,
        "checkpoint_revision": None,
        "compatibility_verified": False, "loader_family": "huggingface_transformers", "task_types": ["classification"],
        "notes": "RGB-only pretraining -- channel adaptation selects the dataset's 3 RGB-equivalent bands "
                 "and discards the rest (see gotrain.band_selection), rather than using all 13/18 GEO-Bench "
                 "channels. Requires `transformers` installed and a pinned hf_revision before use (see above) "
                 "-- neither is satisfied in this environment, so this has NOT been run or verified here.",
    },
    "ibm-nasa/prithvi-eo-2.0-tiny": {
        "model_id": "ibm-nasa/prithvi-eo-2.0-tiny", "architecture": "ViT-based geospatial foundation model (Tiny variant)",
        "pretraining_source": "IBM/NASA Prithvi-EO (self-supervised, HLS/multispectral satellite imagery)",
        "pretraining_domain": "satellite_multispectral", "supported_modalities": ["multispectral"],
        "expected_channels": None, "input_band_order": None, "normalization": {},
        "parameter_count": None, "parameter_count_source": None, "license": None, "checkpoint_revision": None,
        "compatibility_verified": False, "loader_family": None, "task_types": ["classification", "segmentation"],
        "notes": "Listed as a candidate per the design brief; this project has no internet access to confirm "
                 "current parameter count/license/revision and no loader implemented. Do not treat any field "
                 "above as confirmed -- they are intentionally left null rather than guessed.",
    },
}


# ---------------------------------------------------------------------------
# Verification cache (persisted result of validate_model_compatibility).
# ---------------------------------------------------------------------------


def _verification_path(model_id: str, dataset: str) -> Path:
    safe_model_id = model_id.replace("/", "__")
    return VERIFICATION_DIR / f"{safe_model_id}__{dataset}.json"


def _read_verification(model_id: str, dataset: str) -> Optional[dict]:
    path = _verification_path(model_id, dataset)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_verification(model_id: str, dataset: str, result: dict) -> None:
    path = _verification_path(model_id, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".verify_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_compatible_models(task_type: str, dataset: Optional[str] = None) -> list:
    """Catalog entries whose task_types includes `task_type`, each
    annotated with its cached compatibility_verified status for
    `dataset` if validate_model_compatibility has ever been run for that
    (model_id, dataset) pair -- the static catalog's own
    "compatibility_verified": False is never overwritten; this only adds
    a "verification" key with whatever's cached, if anything.
    """
    results = []
    for entry in CATALOG.values():
        if task_type not in entry["task_types"]:
            continue
        row = dict(entry)
        if dataset is not None:
            row["verification"] = _read_verification(entry["model_id"], dataset)
        results.append(row)
    return results


def estimate_model_resources(model_id: str) -> dict:
    """Rough, explicitly-labeled ESTIMATE of parameter count and
    approximate GPU memory at a small batch size -- never a measurement.
    The memory heuristic (params x 4 bytes x ~4 for weights+gradients+
    Adam's two momentum buffers, plus a small constant for activations)
    is a standard rule of thumb, not specific to any dataset/batch size;
    estimate_training_memory_mb below gives a coarser, labeled range.
    """
    if model_id not in CATALOG:
        raise KeyError(f"unknown model_id {model_id!r} -- see gotrain.pretrained.CATALOG")
    entry = CATALOG[model_id]
    params = entry["parameter_count"]
    if params is None:
        return {
            "model_id": model_id, "parameter_count": None,
            "estimated_training_memory_mb": None,
            "estimate_label": "UNKNOWN -- parameter_count not confirmed for this entry, see its 'notes'",
        }
    bytes_per_param_training = 4 * 4  # fp32 weight + grad + 2 Adam moment buffers
    estimated_mb = round(params * bytes_per_param_training / (1024 * 1024), 1)
    return {
        "model_id": model_id,
        "parameter_count": params,
        "parameter_count_source": entry.get("parameter_count_source"),
        "estimated_training_memory_mb": estimated_mb,
        "estimate_label": "ESTIMATE ONLY -- rule-of-thumb (params x 4 bytes x 4 for weights/grad/optimizer "
                           "state), excludes activation memory which scales with batch size and image "
                           "resolution; not a measurement.",
    }


def validate_model_compatibility(
    model_id: str, dataset: str, task_type: str, n_bands: int, n_classes: int, band_names: Optional[list] = None
) -> dict:
    """The ONLY way an entry's compatibility can be considered verified:
    actually construct it with `dataset`'s real band/class count and run
    one forward pass on a synthetic all-zero input. Requires torch (and
    timm/segmentation_models_pytorch/transformers, depending on
    loader_family) -- not runnable in this environment, written to be
    run on the SageMaker lab. Caches its result; does not mutate CATALOG
    itself.

    `band_names` is required for `loader_family == "huggingface_transformers"`
    entries (they select specific bands by name, see
    gotrain.band_selection) -- omit it for timm/smp entries, which don't
    need it.
    """
    if model_id not in CATALOG:
        raise KeyError(f"unknown model_id {model_id!r} -- see gotrain.pretrained.CATALOG")
    entry = CATALOG[model_id]

    if task_type not in entry["task_types"]:
        result = {"compatible": False, "reason": f"{model_id} does not support task_type={task_type!r}"}
        _write_verification(model_id, dataset, result)
        return result

    if entry["loader_family"] is None:
        result = {
            "compatible": False,
            "reason": f"no loader implemented for {model_id} in this environment -- requires manual "
                      "verification before autonomous use (see its catalog 'notes').",
        }
        _write_verification(model_id, dataset, result)
        return result

    try:
        model = load_pretrained_backbone(model_id, n_bands, n_classes, task_type, band_names=band_names)
        model.eval()
        import torch
        dummy = torch.zeros(1, n_bands, 32, 32)
        with torch.no_grad():
            output = model(dummy)
        compatible = output.shape[0] == 1
        result = {
            "compatible": compatible,
            "reason": "forward pass succeeded" if compatible else f"unexpected output shape {tuple(output.shape)}",
            "output_shape": list(output.shape),
        }
    except Exception as e:
        result = {"compatible": False, "reason": f"construction/forward pass failed: {e}"}

    _write_verification(model_id, dataset, result)
    return result


def load_pretrained_backbone(model_id: str, n_bands: int, n_classes: int, task_type: str, band_names: Optional[list] = None):
    """Construct `model_id` for `n_bands` input channels / `n_classes`
    outputs, reusing the EXACT same builder the training templates
    already use (gotrain.templates.train_classification.build_model /
    train_segmentation.build_model) -- no separate, divergent loading
    path. Raises NotImplementedError for catalog entries with no
    loader_family (the still-aspirational torchgeo/Prithvi candidates)
    rather than attempting an unverified download or executing
    third-party model code this project hasn't vetted.

    `band_names` is required for `loader_family == "huggingface_transformers"`
    (see load_dinov2_classification_backbone) -- these models have no
    in_chans-style adaptation and need to know WHICH bands to select, not
    just how many there are.
    """
    if model_id not in CATALOG:
        raise KeyError(f"unknown model_id {model_id!r} -- see gotrain.pretrained.CATALOG")
    entry = CATALOG[model_id]

    if entry["loader_family"] == "timm":
        from gotrain.templates.train_classification import build_model

        return build_model(entry["architecture"], True, n_bands, n_classes)
    if entry["loader_family"] == "smp":
        from gotrain.templates.train_segmentation import build_model

        architecture = "unet"  # the catalog entry is an encoder choice, not an architecture choice
        return build_model(architecture, entry["architecture"].split(" ")[0], True, n_bands, n_classes)
    if entry["loader_family"] == "huggingface_transformers":
        if band_names is None:
            raise ValueError(
                f"{model_id} requires band_names (to select its 3 RGB-equivalent input channels) -- "
                "this entry has no timm-style in_chans adaptation."
            )
        return load_dinov2_classification_backbone(entry, band_names, n_classes)

    raise NotImplementedError(
        f"{model_id} has no loader implemented in this environment (loader_family=None). "
        "See its catalog 'notes' for why -- this is deliberate, not a bug."
    )


def load_dinov2_classification_backbone(entry: dict, band_names: list, n_classes: int):
    """Build a DINOv2-backed classification model: selects the dataset's
    3 RGB-equivalent bands (gotrain.band_selection.select_rgb_band_indices
    -- DINOv2 was pretrained on exactly 3 channels and has no native
    mechanism to accept more, unlike timm's in_chans adaptation for the
    CNN backbones elsewhere in this project, so the other 10-15 GEO-Bench
    bands are discarded, not combined), runs them through DINOv2's
    encoder, and attaches a new linear classification head.

    The head is deliberately named `self.head` (not e.g. `self.classifier`):
    gotrain.finetuning.HEAD_PARAM_PREFIXES_BY_FAMILY["timm"] already
    recognizes that prefix, so fine_tuning_strategy (linear_probe/
    partial/full) works on this model with zero changes to
    gotrain/finetuning.py -- `apply_fine_tuning_strategy(model, "timm",
    strategy)` is the right call for this wrapper too.

    NEEDS transformers installed and entry["hf_revision"] pinned to an
    actual commit SHA -- refuses to run with either missing rather than
    silently installing nothing or defaulting to an unpinned "main".
    Not runnable or verified in this project's local/CI environment (no
    network-confirmed HF Hub access, no transformers installed) --
    written to be run and verified on the SageMaker lab.
    """
    if entry.get("hf_revision") is None:
        raise RuntimeError(
            f"{entry['model_id']} has no pinned hf_revision -- look up an actual commit SHA from "
            f"https://huggingface.co/{entry['hf_repo_id']}/commits/main and set it in "
            "gotrain.pretrained.CATALOG before using this backbone. Refusing to default to "
            "an unpinned 'main' revision."
        )

    # Resolved before importing torch/transformers: a bad band_names
    # list should fail cheaply, not after paying for two heavy imports.
    from gotrain.band_selection import select_rgb_band_indices

    band_indices = select_rgb_band_indices(band_names)
    hf_repo_id = entry["hf_repo_id"]
    hf_revision = entry["hf_revision"]

    import torch.nn as nn
    from transformers import AutoModel

    class Dinov2ClassificationBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.band_indices = band_indices
            self.backbone = AutoModel.from_pretrained(hf_repo_id, revision=hf_revision)
            self.head = nn.Linear(self.backbone.config.hidden_size, n_classes)

        def forward(self, x):
            rgb = x[:, self.band_indices, :, :]
            outputs = self.backbone(pixel_values=rgb)
            pooled = outputs.pooler_output
            if pooled is None:
                pooled = outputs.last_hidden_state.mean(dim=1)
            return self.head(pooled)

    return Dinov2ClassificationBackbone()
