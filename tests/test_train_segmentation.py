"""Tests for gotrain/templates/train_segmentation.py's pure logic.

pad_to_multiple is plain arithmetic (no torch) and is tested directly.
build_model itself (and the padded forward it wraps) needs torch +
segmentation_models_pytorch -- not installed in this environment -- so
its construction/forward-pass behavior is reviewed by inspection, not
executed here. See the module's build_model docstring for why padding
lives there: m-nz-cattle's 500x500 images aren't a multiple of 32 (what
segmentation_models_pytorch's encoders downsample by), which raised a
RuntimeError on the decoder's skip-connection concatenation on the
SageMaker lab; m-pv4ger-seg's 320x320 already is a multiple of 32 and is
confirmed unaffected by this specific bug (the padding amount computed
below is (0, 0) for it) -- its own run failures have a different cause.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("GEO_BENCH_DIR", str(REPO_ROOT / "data" / "geobench"))

from gotrain.templates.train_segmentation import pad_to_multiple


def test_pad_to_multiple_m_nz_cattle_500x500():
    # The actual bug: 500 % 32 == 20, so smp's U-Net decoder's upsampled
    # feature map didn't match its skip connection's size.
    pad_h, pad_w = pad_to_multiple(500, 500)
    assert (pad_h, pad_w) == (12, 12)
    assert (500 + pad_h) % 32 == 0
    assert (500 + pad_w) % 32 == 0


def test_pad_to_multiple_m_pv4ger_seg_320x320_needs_no_padding():
    # 320 is already a multiple of 32 -- confirms this specific bug
    # cannot explain m-pv4ger-seg's run failures.
    assert pad_to_multiple(320, 320) == (0, 0)


@pytest.mark.parametrize("size", [1, 31, 32, 33, 63, 64, 65, 320, 499, 500, 501, 512])
def test_pad_to_multiple_always_reaches_an_exact_multiple(size):
    pad, _ = pad_to_multiple(size, size)
    assert (size + pad) % 32 == 0
    assert 0 <= pad < 32


def test_pad_to_multiple_handles_non_square_images():
    pad_h, pad_w = pad_to_multiple(500, 320)
    assert pad_h == 12
    assert pad_w == 0


def test_pad_to_multiple_respects_custom_multiple():
    assert pad_to_multiple(10, 10, multiple=16) == (6, 6)
    assert pad_to_multiple(16, 16, multiple=16) == (0, 0)
