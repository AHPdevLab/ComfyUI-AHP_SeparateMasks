# ComfyUI-AHP_SeparateMasks

**AHP Separate Masks** splits one mask into a separate mask for each object in it. It produces clean 0/1 edges by default, and it includes Photoshop-style controls (Shift Edge, Smooth, Feather, Contrast) for refining the edge after the cut.

- Node name in ComfyUI: **AHP Separate Masks**
- Category: `AHP/masking`
- Author: **Ali Haghighat panah**
- License: GPL-3.0 (see [LICENSE](LICENSE))

---

## How it works

The node processes each mask in these steps:

1. **Invert (optional).** If `invert_mask` is on, the input becomes `1 − mask` before anything else.
2. **Find objects.** Pixels at or above `detect_threshold` are grouped into connected objects, using 8-way connectivity (diagonal neighbors count as connected). If `bridge_gap` is above 0, gaps up to that many pixels are closed first, so one object split by a thin break stays one object.
3. **Assign every mask pixel to an object.** Each pixel with a mask value goes to its nearest object. The objects never overlap, and together they cover the whole mask.
4. **Cut each object into a 0/1 shape.** Its edge is at the 0.5 level of the input.
5. **Refine the edge.** The shape is passed through `edge_shift`, `smooth`, `feather`, and `contrast`, in that order. With the default settings this step does nothing, so the output stays 0/1.
6. **Filter and sort.** Objects smaller than `size_threshold_width` or `size_threshold_height` are dropped. The rest are sorted left to right by their center.

`detect_threshold` and `bridge_gap` only affect step 2, which is how objects are grouped. They never change the edge of the output.

## Modes

| Mode | What it outputs |
|---|---|
| `area` | Each object's own shape, pixel for pixel |
| `convex_polygons` | The convex hull of the object, simplified to about `max_poly_points` vertices and anti-aliased with `supersample` |
| `box` | A filled rectangle around the object |

## Outputs

| Output | Type | What it contains |
|---|---|---|
| `mask` | MASK | One mask per object. Batch of N objects |
| `frame_index` | AHP_FRAME_INDEX | A list with the source frame of each object, for example `[0, 0, 1]`. Connect it to **AHP Batch Crop By Mask** to crop each object from its own frame |

Objects are ordered by frame first, then left to right inside each frame. Frames with no objects are skipped. If nothing is found at all, the output is one blank mask with frame index `[0]`.

## Inputs

### Required

| Input | Default | What it does |
|---|---|---|
| `mask` | — | The input MASK |
| `invert_mask` | off | Flips the mask (1 − mask) before processing. Turn it on when your objects are the dark area |
| `size_threshold_width` | 256 | Drops objects narrower than this, in pixels |
| `size_threshold_height` | 256 | Drops objects shorter than this, in pixels |
| `mode` | area | `area`, `convex_polygons`, or `box` |
| `max_poly_points` | 8 | Target number of corners for `convex_polygons` mode |

### Optional

| Input | Default | What it does |
|---|---|---|
| `detect_threshold` | 0.5 | Which pixels count as part of an object when grouping. Does not affect the edge |
| `bridge_gap` | 0 | Joins pieces of one object separated by gaps up to this many pixels. 0 = off |
| `edge_shift` | 0.0 | Grows (+) or shrinks (−) each object, in fractional pixels (for example 0.3 or −1.5) |
| `smooth` | 0.0 | Blur radius (float) used to round jagged edges. 0 = off |
| `feather` | 0.0 | Blurs the edge into a soft gradient. Float, step 0.1. 0 = hard edge |
| `hard_edge` | on | On: output is 0/1. Fractional shifts then act on a half-pixel grid. Off: a fractional shift gives a 1px soft edge |
| `contrast` | 0.0 | Sharpens a feathered edge. Has no effect when `feather` is 0 |
| `keep_soft_edges` | off | `area` mode only. Keeps each object's original soft values instead of cutting it to 0/1. Use this to keep the input mask quality. `smooth` is skipped in this mode, since it works on 0/1 shapes |
| `supersample` | 4 | Edge smoothness in `convex_polygons` mode. Higher is smoother and slower |

## Photoshop equivalents

If you know Photoshop's **Select and Mask** (Refine Edge) panel, these controls will feel familiar:

| Photoshop | This node | Notes |
|---|---|---|
| Shift Edge | `edge_shift` | Positive values grow the selection, negative values shrink it |
| Smooth | `smooth` | Removes jagged edges |
| Feather | `feather` | Softens the edge with a blur |
| Contrast | `contrast` | Makes the feathered edge firmer |
| Radius | `feather` | In Photoshop, Radius controls the edge area that gets refined. Here, `feather` is the blur radius |
| Invert | `invert_mask` | Flips the selection |

These are simplified versions of Photoshop's tools. The algorithms are different, so the results won't match Photoshop exactly.

## What's new compared to the original KJNodes SeparateMasks

The original node is the starting point. This version adds the following:

| Area | Original KJNodes | This version |
|---|---|---|
| Invert input | Not available | `invert_mask` switch |
| Grouping control | Fixed at 0.5 | `detect_threshold` |
| Joining split objects | Not available | `bridge_gap` |
| Edge refinement | Not available | `edge_shift`, `smooth`, `feather`, `contrast` (Photoshop-style). `edge_shift`, `smooth`, and `feather` are floats |
| Polygon edges | Stair-stepped | Anti-aliased with `supersample` |
| Output type | 8-bit values (`uint8`) | Float 32-bit values, as ComfyUI expects |
| Empty result | Returns uninitialized memory | Returns zeros with the correct size |
| Input rounding | Input rounded to 0/1 before processing | Input kept as-is; only grouping uses the threshold |
| Console output | Logs every component | No per-component logging |
| Output node flag | `OUTPUT_NODE = True` | `False` (runs only when connected) |
| Category | `KJNodes/masking` | `AHP/masking` |

Both versions output a 0/1 mask by default, so you get the same clean cut-out. The new options are there when you need to change the shape.

## Changes from the original code

The class was taken from `SeparateMasks` in `ComfyUI-KJNodes/nodes/mask_nodes.py` (by kijai, GPL-3.0). The main changes are listed in the table above. In summary:

- Removed the rounding step and the 8-bit output, which caused edge loss.
- Added the invert switch, grouping controls, and the refinement pipeline.
- Replaced the polygon fitting loop with a binary search on the simplification level, and added supersampled rasterization.
- Fixed the empty-result fallback to return zeros of the correct size.
- Renamed the node, category, and files to AHP.

## Installation

Clone into your `ComfyUI/custom_nodes` folder and install the requirements:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-AHP_SeparateMasks.git
pip install -r ComfyUI-AHP_SeparateMasks/requirements.txt
```

Restart ComfyUI. Double-click the canvas and search for **AHP Separate Masks**.

## Tips

- For a clean cut-out, leave everything at its default except `invert_mask` (if needed).
- If one object splits into many pieces, set `bridge_gap` to 1 or 2.
- For a soft edge similar to Photoshop's Feather, try `feather` at 2–6 and `contrast` at 0.3–0.6.
- To grow or shrink an object slightly, use `edge_shift` (for example, +0.5 or −1.5).
- To see exact sub-pixel shifts, turn `hard_edge` off. The boundary then shows partial values equal to the shift's fraction.

## Running the tests

```bash
cd ComfyUI-AHP_SeparateMasks
python tests/test_ahp_separate_masks.py
```

The tests run without ComfyUI. They need `numpy`, `scipy`, `opencv-python`, and `torch`.

## Credits

- **Author and modifications:** Ali Haghighat panah
- **Original code:** `SeparateMasks` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by kijai, licensed under GPL-3.0
- **Development assistance:** Claude (Anthropic), used to help write and test the code

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. Because it's based on GPL-3.0 code, the modified version must also be GPL-3.0. The full text is in [LICENSE](LICENSE).

## Connecting to AHP Batch Crop By Mask

Connect `frame_index` from this node to the `object_frames` input of **AHP Batch Crop By Mask**, and connect `mask` to its `mask` input. Each object is then cropped from the image of the frame it came from. Use `box_mode: per_image` so each object gets its own crop.

To keep the original mask quality through the chain, turn on `keep_soft_edges`.
