"""Band-name utilities for adapting a 3-channel (RGB-pretrained) backbone
to a GEO-Bench dataset's actual multi-band stack.

Needed specifically because DINOv2 (gotrain.pretrained's
"huggingface/facebook-dinov2-small" entry) has no equivalent of timm's
`in_chans` adaptation -- it was pretrained on exactly 3 channels and has
no native mechanism to accept more. select_rgb_band_indices finds which
3 of a dataset's bands are its Red/Green/Blue-equivalent channels, so
those 3 can be selected out and fed to DINOv2 unchanged, discarding the
other bands rather than guessing how to combine them.

Pure string matching -- no torch, no geobench I/O -- fully testable
against this project's own documented band name lists
(data/dataset_summary.md) without needing downloaded data or the ML stack.
"""

from typing import Optional


def _find_standalone_color_band(band_names: list, color: str) -> Optional[int]:
    """Index of the first band whose name contains `color` as a
    standalone word, excluding anything also containing "edge" --
    GEO-Bench's own band names include "Vegetation Red Edge" bands
    (m-eurosat/m-so2sat's red-edge bands), which a plain substring match
    for "red" would wrongly treat as the Red band.
    """
    for i, name in enumerate(band_names):
        words = [w.lower() for w in name.replace("-", " ").replace(".", " ").split()]
        if color in words and "edge" not in words:
            return i
    return None


def select_rgb_band_indices(band_names: list) -> list:
    """Indices of the Red, Green, Blue bands within `band_names`, in
    [red, green, blue] order. Raises ValueError if any of the three
    isn't found as a standalone-named band -- never guesses a fallback
    index, since silently picking the wrong channel is worse than
    refusing to run.
    """
    indices = {}
    for color in ("red", "green", "blue"):
        idx = _find_standalone_color_band(band_names, color)
        if idx is None:
            raise ValueError(
                f"could not find a standalone {color!r} band among {band_names!r} -- "
                "this dataset's bands may not include true-color channels at all."
            )
        indices[color] = idx
    return [indices["red"], indices["green"], indices["blue"]]
