"""Tests for the data-loading time budget (LOADING_PHASE_BUDGET_FRACTION /
DataLoadingBudgetExceeded) in both training templates -- the root-cause fix
for "STOPPED, 0/3, timeout" with no checkpoint, no ledger entry, and no
recorded reason.

Root cause: build_dataset_arrays eagerly loads an entire split into memory
BEFORE training_start is set, so a large split (e.g. m-bigearthnet's ~20k
samples) can alone consume the whole per-run subprocess timeout
(gotrain.tools.training.launch_training's hard subprocess.run(timeout=...)
kill) with zero epochs run and nothing written anywhere. The fix gives the
loading loop its own deadline (derived from the real subprocess timeout,
threaded through as cfg["_timeout_seconds"]) so it raises a clear,
loggable DataLoadingBudgetExceeded with time to spare, instead of being
silently SIGKILLed by the parent process.

Uses a minimal fake geobench-shaped dataset (task_specs.get_dataset(split=...)
-> indexable samples with .pack_to_3d()/.label) rather than the real
geobench package, since what's under test is the deadline-checking loop
itself, not GEO-Bench I/O -- the same kind of fake already used for
dataset_spec in tests/test_dataset_spec.py.
"""

import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.templates import train_classification as tc
from gotrain.templates import train_segmentation as ts


class _FakeSample:
    def __init__(self, label, n_bands=3, h=4, w=4):
        self._label = label
        self._arr = np.zeros((h, w, n_bands), dtype=np.float32)

    def pack_to_3d(self, band_names, **kwargs):
        return self._arr, band_names

    @property
    def label(self):
        return self._label


class _SlowFakeSample(_FakeSample):
    """Mimics a sample whose actual load (disk read + pack_to_3d) takes
    real wall-clock time -- the thing a genuinely large/slow dataset does,
    which is exactly what the deadline check has to catch.
    """

    def __init__(self, label, delay_seconds, **kwargs):
        super().__init__(label, **kwargs)
        self._delay_seconds = delay_seconds

    def pack_to_3d(self, band_names, **kwargs):
        time.sleep(self._delay_seconds)
        return super().pack_to_3d(band_names)


class _FakeSplit:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        return self._samples[i]


class _FakeTaskSpecs:
    def __init__(self, train_samples):
        self._train_samples = train_samples

    def get_dataset(self, split):
        return _FakeSplit(self._train_samples)


_LOGGER = logging.getLogger("test_training_loading_deadline")
_LOGGER.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# train_classification.py
# ---------------------------------------------------------------------------


def test_classification_loading_within_deadline_succeeds():
    task_specs = _FakeTaskSpecs([_FakeSample(label=0) for _ in range(5)])
    deadline = time.time() + 10  # far in the future -- never hits

    images, labels = tc.build_dataset_arrays(
        task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
    )
    assert images.shape[0] == 5
    assert labels.shape == (5,)


def test_classification_loading_raises_when_deadline_already_passed():
    task_specs = _FakeTaskSpecs([_FakeSample(label=0) for _ in range(5)])
    deadline = time.time() - 1  # already in the past

    with pytest.raises(tc.DataLoadingBudgetExceeded) as exc_info:
        tc.build_dataset_arrays(
            task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
        )
    assert "train" in str(exc_info.value)


def test_classification_loading_raises_mid_load_for_a_slow_split():
    """The realistic case: the deadline is reasonable but the split is
    slow enough that it would still be exceeded partway through loading --
    confirms the check fires DURING the loop (via
    LOADING_DEADLINE_CHECK_EVERY), not just as an upfront check.
    """
    n = 300
    samples = [_SlowFakeSample(label=0, delay_seconds=0.01) for _ in range(n)]
    task_specs = _FakeTaskSpecs(samples)
    # Short enough that loading all n samples at 0.01s each cannot finish,
    # long enough that at least one deadline check happens first.
    deadline = time.time() + 0.05

    with pytest.raises(tc.DataLoadingBudgetExceeded):
        tc.build_dataset_arrays(
            task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
        )


def test_classification_loading_with_no_deadline_is_unbounded_as_before():
    """loading_deadline=None (the default) must behave exactly like before
    this fix existed -- no deadline, no exception, however long it takes.
    """
    task_specs = _FakeTaskSpecs([_FakeSample(label=0) for _ in range(3)])
    images, labels = tc.build_dataset_arrays(task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER)
    assert images.shape[0] == 3


def test_classification_loading_respects_max_samples_before_deadline_check():
    """max_samples caps n before the loop even starts -- a tiny max_samples
    with a far-future deadline must simply load that many and succeed,
    not be affected by the deadline logic at all.
    """
    task_specs = _FakeTaskSpecs([_FakeSample(label=0) for _ in range(100)])
    deadline = time.time() + 10
    images, labels = tc.build_dataset_arrays(
        task_specs, "train", ["b1", "b2", "b3"], 5, _LOGGER, loading_deadline=deadline
    )
    assert images.shape[0] == 5


def test_classification_multilabel_loading_also_respects_deadline():
    """The multi-label path (is_multi_label=True, added earlier for
    m-bigearthnet) uses a different label-append branch -- confirms the
    deadline check isn't accidentally single-label-only.
    """
    samples = [_FakeSample(label=np.array([1.0, 0.0, 1.0], dtype=np.float32)) for _ in range(5)]
    task_specs = _FakeTaskSpecs(samples)
    deadline = time.time() - 1

    with pytest.raises(tc.DataLoadingBudgetExceeded):
        tc.build_dataset_arrays(
            task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER,
            is_multi_label=True, loading_deadline=deadline,
        )


# ---------------------------------------------------------------------------
# train_segmentation.py
# ---------------------------------------------------------------------------


def _fake_seg_sample(h=4, w=4, n_bands=3):
    sample = _FakeSample(label=None, n_bands=n_bands, h=h, w=w)
    sample._label = SimpleNamespace(data=np.zeros((h, w), dtype=np.int64))
    return sample


def test_segmentation_loading_within_deadline_succeeds():
    task_specs = _FakeTaskSpecs([_fake_seg_sample() for _ in range(5)])
    deadline = time.time() + 10

    images, masks = ts.build_dataset_arrays(
        task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
    )
    assert images.shape[0] == 5
    assert masks.shape[0] == 5


def test_segmentation_loading_raises_when_deadline_already_passed():
    task_specs = _FakeTaskSpecs([_fake_seg_sample() for _ in range(5)])
    deadline = time.time() - 1

    with pytest.raises(ts.DataLoadingBudgetExceeded):
        ts.build_dataset_arrays(
            task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
        )


def test_segmentation_loading_with_no_deadline_is_unbounded_as_before():
    task_specs = _FakeTaskSpecs([_fake_seg_sample() for _ in range(3)])
    images, masks = ts.build_dataset_arrays(task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER)
    assert images.shape[0] == 3


def test_segmentation_loading_raises_mid_load_for_a_slow_split():
    n = 300
    samples = []
    for _ in range(n):
        s = _SlowFakeSample(label=None, delay_seconds=0.01)
        s._label = SimpleNamespace(data=np.zeros((4, 4), dtype=np.int64))
        samples.append(s)
    task_specs = _FakeTaskSpecs(samples)
    deadline = time.time() + 0.05

    with pytest.raises(ts.DataLoadingBudgetExceeded):
        ts.build_dataset_arrays(
            task_specs, "train", ["b1", "b2", "b3"], None, _LOGGER, loading_deadline=deadline
        )
