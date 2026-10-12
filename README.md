# ComfyUI-AHP_SeparateMasks

A ComfyUI custom node that splits a mask into one clean mask per object. The output is a hard 0/1 mask by default, and you can soften or refine the edge with the options below.

Node name in ComfyUI: **AHP Separate Masks** (category `AHP/masking`)

## Modes

| Mode | Output |
|------|--------|
| `area` | Each object's own shape |
| `convex_polygons` | Convex hull with about `max_poly_points` vertices |
| `box` | Filled bounding box |

## Inputs

### Required

| Name | Default | Description |
|------|---------|-------------|
| `mask` | — | Input MASK |
| `invert_mask` | off | Flips the input (1 − mask) before processing |
| `size_threshold_width` | 256 | Ignore objects narrower than this (pixels) |
| `size_threshold_height` | 256 | Ignore objects shorter than this (pixels) |
| `mode` | area | `area`, `convex_polygons`, or `box` |
| `max_poly_points` | 8 | Vertex target for convex_polygons mode |

### Optional

| Name | Default | Description |
|------|---------|-------------|
| `detect_threshold` | 0.5 | Decides which pixels are grouped into one object. Does not change the edge |
| `bridge_gap` | 0 | Joins pieces of one object separated by gaps up to this many pixels. 0 = off |
| `edge_shift` | 0 | Grows (+) or shrinks (−) each object by this many pixels |
| `smooth` | 0 | Rounds jagged edges. Output stays binary. 0 = off |
| `feather` | 0.0 | Blur radius that softens the edge. 0 = hard edge |
| `contrast` | 0.0 | Sharpens the feathered edge. Has no effect when `feather` is 0 |
| `supersample` | 4 | Edge smoothness for convex_polygons mode |

## Processing order

1. `invert_mask`
2. `detect_threshold` and `bridge_gap` decide which pixels form each object
3. Each object is cut out as a hard 0/1 shape
4. `edge_shift`
5. `smooth`
6. `feather`
7. `contrast`

Outputs are sorted left to right by the center of each object.

## Tips

- For a clean cut-out, leave everything except `invert_mask` at its default.
- If one object splits into many pieces, raise `bridge_gap` to 1 or 2.
- For a soft edge like Photoshop's Feather, set `feather` to 2–6 and `contrast` to 0.3–0.6.

## Installation

Clone into your `ComfyUI/custom_nodes` folder and install the requirements:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-AHP_SeparateMasks.git
pip install -r ComfyUI-AHP_SeparateMasks/requirements.txt
```

Restart ComfyUI.

## Running the tests

```bash
cd ComfyUI-AHP_SeparateMasks
python tests/test_ahp_separate_masks.py
```

## Credits

Based on `SeparateMasks` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai, which is GPL-3.0. This project is a modified version and is released under the same license.

## License

GPL-3.0. See [LICENSE](LICENSE).
