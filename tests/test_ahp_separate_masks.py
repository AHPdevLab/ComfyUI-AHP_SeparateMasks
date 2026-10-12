"""
Run from the package folder:   python -m tests.test_ah_separate_masks
or with pytest:                python -m pytest tests -q
"""

import importlib.util
import os
import sys
import types

import cv2
import numpy as np
import torch

# Outside ComfyUI, `comfy.utils` does not exist. Provide a no-op ProgressBar so the tests run.
try:
    from comfy.utils import ProgressBar  # noqa: F401
except ImportError:
    _comfy = types.ModuleType("comfy")
    _utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, n):
            pass

    _utils.ProgressBar = ProgressBar
    _comfy.utils = _utils
    sys.modules["comfy"] = _comfy
    sys.modules["comfy.utils"] = _utils

PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "ah_pkg", os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
)
pkg = importlib.util.module_from_spec(_spec)
sys.modules["ah_pkg"] = pkg
_spec.loader.exec_module(pkg)

NODE = pkg.NODE_CLASS_MAPPINGS["AHP_SeparateMasks"]()


def blob_mask(H=128, W=128, rects=((20, 20, 60, 60),), aa=True):
    """Rectangles given as (x0, y0, x1_exclusive, y1_exclusive). aa=True adds soft edges."""
    m = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 1.0
    if aa:
        m = cv2.GaussianBlur(m, (0, 0), 1.0)
    return torch.from_numpy(m)[None]


def fractional(a):
    return int(((a > 0) & (a < 1)).sum())


# ---------------------------------------------------------------- registration


def test_registration():
    assert "AHP_SeparateMasks" in pkg.NODE_CLASS_MAPPINGS
    assert pkg.NODE_DISPLAY_NAME_MAPPINGS["AHP_SeparateMasks"] == "AHP Separate Masks"
    inputs = NODE.INPUT_TYPES()
    assert set(inputs["required"]["mode"][0]) == {"convex_polygons", "area", "box"}


# ---------------------------------------------------------------- area mode


def test_area_output_shape_and_dtype():
    (out,) = NODE.separate(blob_mask(), 1, 1, 8, "area")
    assert out.shape == (1, 128, 128)
    assert out.dtype == torch.float32


def test_area_preserves_soft_edges():
    m = blob_mask()
    (out,) = NODE.separate(m, 1, 1, 8, "area")
    o = out[0].numpy()
    soft = m[0].numpy()
    assert fractional(o) > 0, "anti-aliased values were lost"
    kept = o > 0
    assert np.allclose(o[kept], soft[kept]), (
        "kept pixels must match the original soft values"
    )


def test_area_two_components_sorted_left_to_right():
    m = blob_mask(
        rects=((70, 20, 100, 60), (10, 20, 40, 60))
    )  # right blob listed first
    (out,) = NODE.separate(m, 1, 1, 8, "area")
    assert out.shape[0] == 2
    cx = [np.where(out[i].numpy() > 0.5)[1].mean() for i in range(2)]
    assert cx[0] < cx[1]


def test_size_threshold_filters_small_components():
    m = blob_mask(rects=((10, 10, 14, 14), (40, 40, 100, 100)))
    (out,) = NODE.separate(m, 20, 20, 8, "area")
    assert out.shape[0] == 1


def test_edge_reach_keeps_faint_pixel_next_to_object():
    m = np.zeros((64, 64), np.float32)
    m[20:40, 20:40] = 1.0
    m[19, 25] = 0.3  # one pixel above the object
    (out,) = NODE.separate(torch.from_numpy(m)[None], 1, 1, 8, "area")
    assert abs(out[0, 19, 25].item() - 0.3) < 1e-6


def test_edge_reach_ignores_distant_haze():
    m = np.zeros((64, 64), np.float32)
    m[20:40, 20:40] = 1.0
    m[5, 60] = 0.2  # faint speck far from the object
    (out,) = NODE.separate(torch.from_numpy(m)[None], 1, 1, 8, "area")
    assert out[0, 5, 60].item() == 0.0


# ---------------------------------------------------------------- box mode


def test_box_mode_fills_bounding_box():
    m = blob_mask(rects=((30, 30, 70, 60),), aa=False)
    (out,) = NODE.separate(m, 1, 1, 8, "box")
    o = out[0].numpy()
    ys, xs = np.where(o > 0)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (30, 59, 30, 69)
    assert set(np.unique(o).tolist()) <= {0.0, 1.0}


def test_feather_softens_box_edges():
    m = blob_mask(rects=((30, 30, 70, 70),), aa=False)
    (hard,) = NODE.separate(m, 1, 1, 8, "box")
    (soft,) = NODE.separate(m, 1, 1, 8, "box", feather=2.0)
    assert fractional(hard[0].numpy()) == 0
    assert fractional(soft[0].numpy()) > 0


# ---------------------------------------------------------------- polygon mode


def test_polygon_mode_is_antialiased_with_correct_area():
    m = blob_mask(rects=((30, 30, 90, 90),))
    (out,) = NODE.separate(m, 1, 1, 8, "convex_polygons")
    o = out[0].numpy()
    assert o.dtype == np.float32
    assert fractional(o) > 0, "polygon edges should be anti-aliased"
    expected = 3540.0  # roughly a 60x60 square
    assert abs(o.sum() - expected) / expected < 0.05


# ---------------------------------------------------------------- batch and empty input


def test_empty_mask_returns_zeros_of_correct_size():
    m = torch.zeros(1, 32, 40)
    (out,) = NODE.separate(m, 1, 1, 8, "area")
    assert out.shape == (1, 32, 40)
    assert out.abs().sum().item() == 0.0


def test_batch_input_collects_components_from_all_frames():
    m = torch.stack(
        [
            blob_mask()[0],
            blob_mask(rects=((10, 10, 50, 50), (70, 70, 110, 110)))[0],
        ]
    )
    (out,) = NODE.separate(m, 1, 1, 8, "area")
    assert out.shape == (3, 128, 128)


def test_no_rounding_of_input_values():
    m = blob_mask()
    (out,) = NODE.separate(m, 1, 1, 8, "area")
    assert fractional(out[0].numpy()) > 0


if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
