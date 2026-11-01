"""Tests for gotrain/stresslab/perturbations.py. Pure numpy, synthetic
arrays -- no torch, no geobench I/O needed.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.stresslab import perturbations


def _synthetic_images(n=5, h=8, w=8, c=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=100.0, scale=20.0, size=(n, h, w, c)).astype(np.float32)


def test_add_gaussian_noise_preserves_shape_and_dtype():
    images = _synthetic_images()
    out = perturbations.add_gaussian_noise(images, "mild")
    assert out.shape == images.shape
    assert out.dtype == np.float32


def test_gaussian_noise_magnitude_increases_with_severity():
    images = _synthetic_images()
    mild = perturbations.add_gaussian_noise(images, "mild", seed=1)
    moderate = perturbations.add_gaussian_noise(images, "moderate", seed=1)
    severe = perturbations.add_gaussian_noise(images, "severe", seed=1)

    mild_delta = np.abs(mild.astype(np.float64) - images.astype(np.float64)).mean()
    moderate_delta = np.abs(moderate.astype(np.float64) - images.astype(np.float64)).mean()
    severe_delta = np.abs(severe.astype(np.float64) - images.astype(np.float64)).mean()
    assert mild_delta < moderate_delta < severe_delta


def test_gaussian_noise_is_relative_to_band_scale_not_absolute():
    # A band with std=0 everywhere (constant channel) should stay
    # essentially unperturbed -- noise_std scales with the band's own
    # observed std, so a flat channel gets ~zero noise, not a fixed
    # absolute jolt.
    images = np.zeros((5, 8, 8, 1), dtype=np.float32)
    out = perturbations.add_gaussian_noise(images, "severe")
    assert np.allclose(out, 0.0)


def test_dropout_bands_zeroes_exactly_n_bands():
    images = _synthetic_images(c=5)
    out = perturbations.dropout_bands(images, "moderate", seed=2)
    zeroed_bands = [c for c in range(5) if np.all(out[..., c] == 0.0)]
    assert len(zeroed_bands) == 2


def test_dropout_bands_never_zeroes_every_band_even_at_severe():
    images = _synthetic_images(c=3)  # e.g. m-pv4ger-seg/m-nz-cattle's plain RGB
    out = perturbations.dropout_bands(images, "severe", seed=3)
    zeroed_bands = [c for c in range(3) if np.all(out[..., c] == 0.0)]
    assert len(zeroed_bands) <= 2  # at least 1 of 3 bands survives


def test_dropout_bands_is_deterministic_for_a_given_seed():
    images = _synthetic_images(c=5)
    out1 = perturbations.dropout_bands(images, "mild", seed=7)
    out2 = perturbations.dropout_bands(images, "mild", seed=7)
    assert np.array_equal(out1, out2)


def test_shift_brightness_magnitude_increases_with_severity():
    images = _synthetic_images()
    mild = perturbations.shift_brightness(images, "mild")
    severe = perturbations.shift_brightness(images, "severe")
    mild_delta = (mild.astype(np.float64) - images.astype(np.float64)).mean()
    severe_delta = (severe.astype(np.float64) - images.astype(np.float64)).mean()
    assert abs(mild_delta) < abs(severe_delta)


def test_horizontal_flip_reverses_width_axis():
    images = np.arange(2 * 1 * 4 * 1, dtype=np.float32).reshape(2, 1, 4, 1)
    out = perturbations.horizontal_flip(images)
    assert np.array_equal(out[0, 0, :, 0], images[0, 0, ::-1, 0])


def test_rotate_90_rotates_a_known_array():
    images = np.arange(1 * 2 * 2 * 1, dtype=np.float32).reshape(1, 2, 2, 1)
    out = perturbations.rotate_90(images)
    expected = np.rot90(images[0, :, :, 0], k=1)
    assert np.array_equal(out[0, :, :, 0], expected)


def test_perturbations_registry_and_severities_are_consistent():
    assert set(perturbations.PERTURBATIONS) == set(perturbations.SEVERITIES_FOR)
    for name, severities in perturbations.SEVERITIES_FOR.items():
        fn = perturbations.PERTURBATIONS[name]
        for severity in severities:
            out = fn(_synthetic_images(), severity)
            assert out.shape == _synthetic_images().shape


@pytest.mark.parametrize("severity", perturbations.SEVERITIES)
def test_all_scaled_perturbations_accept_every_severity(severity):
    images = _synthetic_images()
    for name in ("gaussian_noise", "band_dropout", "brightness_shift"):
        fn = perturbations.PERTURBATIONS[name]
        out = fn(images, severity)
        assert out.shape == images.shape
