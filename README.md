# ComfyUI-AHP_SeparateMasks

A ComfyUI custom node that splits a mask into one mask per object, keeping the original soft (anti-aliased) edges.

Node name in ComfyUI: **AHP Separate Masks** (category `AHP/masking`)

## Modes

| Mode | Output |
|------|--------|
| `area` | Each object's original mask values, including anti-aliased edges. Together the outputs reproduce the input mask |
| `convex_polygons` | Anti-aliased convex hull with about `max_poly_points` vertices |
| `box` | Filled bounding box (hard edges unless `feather` is above 0) |

## Inputs

| Name | Default | Description |
|------|---------|-------------|
| `mask` | — | Input MASK |
| `size_threshold_width` | 256 | Ignore objects narrower than this (pixels) |
| `size_threshold_height` | 256 | Ignore objects shorter than this (pixels) |
| `mode` | convex_polygons | `convex_polygons`, `area`, or `box` |
| `max_poly_points` | 8 | Vertex target for polygon mode |
| `detect_threshold` | 0.5 | Decides how pixels are grouped into objects. Does not change the output edges |
| `bridge_gap` | 0 | Joins pieces of one object separated by gaps up to about this many pixels. 0 = off |
| `feather` | 0.0 | Blur radius for each output. 0 = off |
| `supersample` | 4 | Anti-aliasing quality for polygon mode |

Outputs are sorted left to right by the center of each object. Output is a float MASK with values from 0 to 1.

## Tips

- If one object splits into many pieces, raise `bridge_gap` to 1 or 2.
- `detect_threshold` only affects how objects are found. Use the default unless you need to separate faint objects.

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
python tests/test_ah_separate_masks.py
```

## Credits

Based on `SeparateMasks` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai, which is GPL-3.0. This project is a modified version and is released under the same license.

## License

GPL-3.0. See [LICENSE](LICENSE).
