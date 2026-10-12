"""
Run from the package folder:   python tests/test_ahp_separate_masks.py
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
    "ahp_pkg", os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
)
pkg = importlib.util.module_from_spec(_spec)
sys.modules["ahp_pkg"] = pkg
_spec.loader.exec_module(pkg)

NODE = pkg.NODE_CLASS_MAPPINGS["AHP_SeparateMasks"]()


def run(mask, w=1, h=1, mode="area", **kw):
    """Call the node with ComfyUI-style keyword arguments and return the output tensor."""
    out, = NODE.separate(mask, size_threshold_width=w, size_threshold_height=h,
                         max_poly_points=8, mode=mode, **kw)
    return out


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


def is_binary(a):
    return set(np.unique(a).tolist()) <= {0.0, 1.0}


def maze_mask(H=96, W=96, wall=3, gap=2):
    """A comb-like shape: one spine with thin teeth. Teeth are separated from the spine by `gap` px."""
    m = np.zeros((H, W), np.float32)
    m[10:10 + wall, 5:W - 5] = 1.0  # spine
    for x in range(8, W - 8, 10):
        m[10 + wall + gap:H - 10, x:x + wall] = 1.0  # teeth, not touching the spine
    return torch.from_numpy(cv2.GaussianBlur(m, (0, 0), 0.7))[None]


# ---------------------------------------------------------------- registration

def test_registration():
    assert "AHP_SeparateMasks" in pkg.NODE_CLASS_MAPPINGS
    assert pkg.NODE_DISPLAY_NAME_MAPPINGS["AHP_SeparateMasks"] == "AHP Separate Masks"
    assert NODE.CATEGORY == "AHP/masking"
    inputs = NODE.INPUT_TYPES()
    assert set(inputs["required"]["mode"][0]) == {"area", "convex_polygons", "box"}
    assert inputs["required"]["invert_mask"][0] == "BOOLEAN"
    assert inputs["required"]["invert_mask"][1]["default"] is False
    for name in ("detect_threshold", "bridge_gap", "edge_shift", "smooth", "feather", "contrast", "supersample"):
        assert name in inputs["optional"], f"missing option {name}"


# ---------------------------------------------------------------- output is clean 0/1

def test_default_output_is_binary():
    out = run(blob_mask(), mode="area")
    assert is_binary(out[0].numpy())


def test_default_polygon_output_is_binary():
    out = run(blob_mask(rects=((30, 30, 90, 90),)), mode="convex_polygons")
    assert is_binary(out[0].numpy())


def test_default_box_output_is_binary():
    out = run(blob_mask(rects=((30, 30, 70, 60),)), mode="box")
    assert is_binary(out[0].numpy())


def test_area_output_shape_and_dtype():
    out = run(blob_mask())
    assert out.shape == (1, 128, 128)
    assert out.dtype == torch.float32


def test_area_matches_thresholded_input():
    m = blob_mask()
    out = run(m)
    expected = (m[0].numpy() >= 0.5).astype(np.float32)
    assert np.array_equal(out[0].numpy(), expected)


def test_no_overlap_between_objects():
    m = blob_mask(rects=((10, 10, 40, 40), (50, 50, 100, 100)))
    out = run(m)
    assert out.shape[0] == 2
    assert (out.sum(dim=0) <= 1).all()


# ---------------------------------------------------------------- invert switch

def test_invert_switch_flips_input_first():
    m = blob_mask()
    normal = run(m, invert_mask=False)
    inverted_input = run(1.0 - m, invert_mask=True)
    assert torch.equal(normal, inverted_input)


def test_invert_switch_off_is_unchanged():
    m = blob_mask()
    assert torch.equal(run(m, invert_mask=False), run(m))


# ---------------------------------------------------------------- grouping

def test_three_objects_give_three_masks():
    m = blob_mask(rects=((10, 10, 40, 40), (60, 10, 90, 40), (30, 70, 80, 110)))
    assert run(m).shape[0] == 3


def test_two_components_sorted_left_to_right():
    m = blob_mask(rects=((70, 20, 100, 60), (10, 20, 40, 60)))  # right blob listed first
    out = run(m)
    assert out.shape[0] == 2
    cx = [np.where(out[i].numpy() > 0.5)[1].mean() for i in range(2)]
    assert cx[0] < cx[1]


def test_threshold_does_not_change_area_output():
    m = blob_mask()
    base = run(m, detect_threshold=0.5)
    for thr in (0.2, 0.35, 0.65, 0.8):
        assert torch.equal(run(m, detect_threshold=thr), base), f"threshold {thr} changed the mask"


def test_size_threshold_filters_small_components():
    m = blob_mask(rects=((10, 10, 14, 14), (40, 40, 100, 100)))
    assert run(m, w=20, h=20).shape[0] == 1


def test_maze_teeth_with_gaps_split_by_default_and_join_with_bridge():
    m = maze_mask(gap=2)  # teeth sit 2 px away from the spine
    default = run(m)
    bridged = run(m, bridge_gap=1)
    assert default.shape[0] > 1
    assert bridged.shape[0] == 1, f"bridged maze still split into {bridged.shape[0]} pieces"


def test_connected_maze_stays_one_object():
    m = maze_mask(gap=0)  # teeth touch the spine
    assert run(m).shape[0] == 1


def test_bridge_gap_does_not_change_output_values():
    m = blob_mask(rects=((20, 20, 50, 60), (52, 20, 80, 60)))
    assert run(m, bridge_gap=1).sum().item() == run(m).sum().item()


# ---------------------------------------------------------------- edge refinement

def test_edge_shift_grows_and_shrinks():
    m = blob_mask()
    base = run(m).sum().item()
    grown = run(m, edge_shift=2).sum().item()
    shrunk = run(m, edge_shift=-2).sum().item()
    assert grown > base > shrunk


def test_edge_shift_keeps_binary_output():
    out = run(blob_mask(), edge_shift=3)
    assert is_binary(out[0].numpy())


def test_smooth_keeps_binary_output():
    out = run(blob_mask(rects=((20, 20, 60, 60),)), smooth=3)
    assert is_binary(out[0].numpy())


def test_feather_creates_soft_edge():
    out = run(blob_mask(), feather=2.0)
    assert fractional(out[0].numpy()) > 0


def test_no_feather_means_no_soft_pixels():
    out = run(blob_mask(), feather=0.0, contrast=1.0)
    assert fractional(out[0].numpy()) == 0


def test_contrast_tightens_feathered_edge():
    m = blob_mask()
    soft = run(m, feather=2.0, contrast=0.0)
    tight = run(m, feather=2.0, contrast=1.0)
    assert fractional(tight[0].numpy()) < fractional(soft[0].numpy())


def test_feather_is_the_only_source_of_soft_values():
    out = run(blob_mask(rects=((20, 20, 60, 60),)), edge_shift=1, smooth=2, contrast=0.5)
    assert is_binary(out[0].numpy())


# ---------------------------------------------------------------- box and polygon modes

def test_box_mode_fills_bounding_box():
    m = blob_mask(rects=((30, 30, 70, 60),), aa=False)
    o = run(m, mode="box")[0].numpy()
    ys, xs = np.where(o > 0)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (30, 59, 30, 69)


def test_polygon_mode_area_is_close_to_square():
    m = blob_mask(rects=((30, 30, 90, 90),))
    o = run(m, mode="convex_polygons")[0].numpy()
    expected = 3540.0  # roughly a 60x60 square
    assert abs(o.sum() - expected) / expected < 0.05


# ---------------------------------------------------------------- batch and empty input

def test_empty_mask_returns_zeros_of_correct_size():
    out = run(torch.zeros(1, 32, 40))
    assert out.shape == (1, 32, 40)
    assert out.abs().sum().item() == 0.0


def test_batch_input_collects_objects_from_all_frames():
    m = torch.stack([
        blob_mask()[0],
        blob_mask(rects=((10, 10, 50, 50), (70, 70, 110, 110)))[0],
    ])
    assert run(m).shape == (3, 128, 128)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
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
