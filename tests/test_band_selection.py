"""Tests for gotrain/band_selection.py, using the EXACT verbatim band name
strings documented in data/dataset_summary.md for all 4 datasets -- not
synthetic examples. Pure string matching, no torch/geobench I/O needed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.band_selection import select_rgb_band_indices

# Copied verbatim from data/dataset_summary.md.
M_EUROSAT_BANDS = [
    "01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "09 - Water vapour",
    "10 - SWIR - Cirrus", "11 - SWIR", "12 - SWIR",
]

M_SO2SAT_BANDS = [
    "01 - VH.Real", "02 - VH.Imaginary", "03 - VV.Real", "04 - VV.Imaginary",
    "05 - VH.LEE Filtered", "06 - VV.LEE Filtered", "07 - VH.LEE Filtered.Real",
    "08 - VV.LEE Filtered.Imaginary", "02 - Blue", "03 - Green", "04 - Red",
    "05 - Vegetation Red Edge", "06 - Vegetation Red Edge", "07 - Vegetation Red Edge",
    "08 - NIR", "08A - Vegetation Red Edge", "11 - SWIR", "12 - SWIR",
]

M_PV4GER_SEG_BANDS = ["Red", "Green", "Blue"]
M_NZ_CATTLE_BANDS = ["Red", "Green", "Blue"]


def test_m_eurosat_finds_correct_rgb_indices():
    # Band 04 (index 3) is Red; the "Vegetation Red Edge" bands at
    # indices 4/5/6/8 must NOT be mistaken for it.
    assert select_rgb_band_indices(M_EUROSAT_BANDS) == [3, 2, 1]


def test_m_so2sat_finds_optical_rgb_not_sar_bands():
    # The optical Red/Green/Blue sit at indices 10/9/8 -- the 8 leading
    # SAR bands (VH.Real, VV.Imaginary, etc.) must never be selected,
    # and "VH.Real"/"VV.Real" containing "real" must not be confused
    # with "red".
    assert select_rgb_band_indices(M_SO2SAT_BANDS) == [10, 9, 8]


def test_m_pv4ger_seg_plain_rgb_bands():
    assert select_rgb_band_indices(M_PV4GER_SEG_BANDS) == [0, 1, 2]


def test_m_nz_cattle_plain_rgb_bands():
    assert select_rgb_band_indices(M_NZ_CATTLE_BANDS) == [0, 1, 2]


def test_vegetation_red_edge_alone_is_not_mistaken_for_red():
    # If Red Edge bands were the ONLY "red"-containing bands, there is
    # genuinely no Red band -- must raise, never fall back to a wrong one.
    with pytest.raises(ValueError):
        select_rgb_band_indices(["Blue", "Green", "Vegetation Red Edge"])


def test_raises_when_a_color_is_entirely_missing():
    with pytest.raises(ValueError) as exc_info:
        select_rgb_band_indices(["Blue", "Green"])
    assert "red" in str(exc_info.value).lower()


def test_case_insensitive_matching():
    assert select_rgb_band_indices(["RED", "green", "BlUe"]) == [0, 1, 2]
