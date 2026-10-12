# AHP Separate Masks
# Copyright (C) 2026 Ali Haghighat Panah
#
# This file is a modified version of SeparateMasks from ComfyUI-KJNodes
# (https://github.com/kijai/ComfyUI-KJNodes, nodes/mask_nodes.py), copyright kijai,
# licensed under GPL-3.0. All changes and new features were made by Ali Haghighat Panah.
# Development was assisted by Claude (Anthropic).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Modified from the original: see README.md for the list of changes.

from inspect import cleandoc

import cv2
import numpy as np
import scipy.ndimage
import torch
from comfy.utils import ProgressBar


class AHP_SeparateMasks:
    """
    Separates a mask into one mask per object.

    Output is a clean binary mask (only 0 and 1) by default. Use the refinement
    options to change the edge, the same way you would in Photoshop.

    Processing order
    ----------------
    1. invert_mask: flips the input (1 - mask) before anything else.
    2. detect_threshold and bridge_gap: decide which pixels belong to the same object.
    3. Each object is cut out as a hard 0/1 shape.
    4. edge_shift: grows (+) or shrinks (-) the shape in pixels.
    5. smooth: rounds jagged edges while keeping the shape binary.
    6. feather: blurs the edge into a soft gradient.
    7. contrast: pushes the soft edge back toward 0 and 1.

    Modes
    -----
    area:
        The object's own shape, pixel for pixel.
    convex_polygons:
        Convex hull with about `max_poly_points` vertices.
    box:
        Filled bounding box.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. Here it is "separate", so the node runs AHP_SeparateMasks().separate()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    separate(...) -> tuple:
        The entry point method. Its name must be the same as the value of property `FUNCTION`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional.
            - Value input_fields (`dict`): Contains input fields config.
        """
        return {
            "required": {
                "mask": (
                    "MASK",
                    {"tooltip": "The mask to split into one mask per object."},
                ),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Invert the input mask (1 - mask) before processing.",
                    },
                ),
                "size_threshold_width": (
                    "INT",
                    {
                        "default": 256,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Ignore objects narrower than this many pixels.",
                    },
                ),
                "size_threshold_height": (
                    "INT",
                    {
                        "default": 256,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Ignore objects shorter than this many pixels.",
                    },
                ),
                "mode": (
                    ["area", "convex_polygons", "box"],
                    {
                        "tooltip": "area keeps the object's own shape. convex_polygons gives a simplified outline. box gives a bounding box.",
                    },
                ),
                "max_poly_points": (
                    "INT",
                    {
                        "default": 8,
                        "min": 3,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Target number of vertices in convex_polygons mode.",
                    },
                ),
            },
            "optional": {
                "detect_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                        "tooltip": "Decides which pixels are grouped into objects. Does not change the edge of the output.",
                    },
                ),
                "bridge_gap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Joins pieces of one object separated by gaps up to about this many pixels. Use it if one object splits into many parts. 0 = off.",
                    },
                ),
                "edge_shift": (
                    "INT",
                    {
                        "default": 0,
                        "min": -16,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Grows (positive) or shrinks (negative) each object by this many pixels.",
                    },
                ),
                "smooth": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Rounds jagged edges. Output stays binary. 0 = off.",
                    },
                ),
                "feather": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.5,
                        "round": 0.001,
                        "display": "number",
                        "tooltip": "Blur radius that softens the edge. 0 = hard 0/1 edge.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "slider",
                        "tooltip": "Sharpens the feathered edge. 0 = off. Has no effect when feather is 0.",
                    },
                ),
                "supersample": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Edge smoothness for convex_polygons mode. Higher is smoother but slower.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "separate"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",)  # Tooltips for the output node

    CATEGORY = "AHP/masking"

    def polygon_to_mask(self, polygon, shape, supersample=4):
        """Rasterize a polygon with supersampling. Returns a binary 0/1 mask with smoother edges."""
        H, W = shape
        s = max(1, int(supersample))
        pts = np.round((polygon.astype(np.float64) + 0.5) * s - 0.5).astype(np.int32)
        big = np.zeros((H * s, W * s), dtype=np.uint8)
        cv2.fillPoly(big, [pts], 255)
        small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
        return (small >= 128).astype(np.float32)

    def get_mask_polygon(self, mask_np, max_points):
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        perimeter = cv2.arcLength(hull, True)

        # Binary search on epsilon to get as close as possible to max_points.
        lo, hi = perimeter * 0.0005, perimeter * 0.2
        best, best_diff = hull, float("inf")
        for _ in range(30):
            eps = (lo + hi) / 2
            approx = cv2.approxPolyDP(hull, eps, True)
            n = len(approx)
            if abs(n - max_points) < best_diff:
                best, best_diff = approx, abs(n - max_points)
            if n == max_points:
                break
            if n > max_points:
                lo = eps  # too many points: simplify more
            else:
                hi = eps  # too few points: simplify less
        return best.reshape(-1, 2)

    def refine(self, shape, edge_shift, smooth, feather, contrast):
        """Photoshop-style edge refinement. `shape` is a binary float32 array."""
        m = shape.astype(np.uint8)

        if edge_shift != 0:
            k = 2 * abs(int(edge_shift)) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.dilate(m, kernel) if edge_shift > 0 else cv2.erode(m, kernel)

        out = m.astype(np.float32)

        if smooth > 0:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=float(smooth))
            out = (blurred >= 0.5).astype(np.float32)

        if feather > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(feather))

        if contrast > 0 and feather > 0:
            gain = 1.0 + 10.0 * float(contrast)
            out = np.clip((out - 0.5) * gain + 0.5, 0.0, 1.0)

        return out.astype(np.float32)

    def separate(
        self,
        mask: torch.Tensor,
        size_threshold_width: int,
        size_threshold_height: int,
        max_poly_points: int,
        mode: str,
        invert_mask: bool = False,
        detect_threshold: float = 0.5,
        bridge_gap: int = 0,
        edge_shift: int = 0,
        smooth: int = 0,
        feather: float = 0.0,
        contrast: float = 0.0,
        supersample: int = 4,
    ):
        B, H, W = mask.shape
        soft_all = mask.detach().float().clamp(0, 1).cpu().numpy()
        if invert_mask:
            soft_all = 1.0 - soft_all
        structure = np.ones((3, 3), dtype=np.uint8)
        separated = []

        for b in range(B):
            soft = soft_all[b]

            # Step 1: decide which pixels form objects. The threshold is used only here.
            binary = soft >= detect_threshold
            if bridge_gap > 0:
                binary = scipy.ndimage.binary_closing(
                    binary, structure=structure, iterations=int(bridge_gap)
                )
            labeled, ncomponents = scipy.ndimage.label(
                binary.astype(np.uint8), structure=structure
            )
            if ncomponents == 0:
                continue

            # Step 2: each pixel with a mask value belongs to its nearest object,
            # so objects never overlap and their union covers the whole mask.
            if (labeled == 0).any():
                _, (iy, ix) = scipy.ndimage.distance_transform_edt(
                    labeled == 0, return_indices=True
                )
                owner = labeled[iy, ix]
            else:
                owner = labeled.copy()
            owner = np.where(soft > 0, owner, 0)

            pbar = ProgressBar(ncomponents)
            for comp in range(1, ncomponents + 1):
                comp_bin = labeled == comp
                rows = np.any(comp_bin, axis=1)
                cols = np.any(comp_bin, axis=0)
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                centroid_x = (x_min + x_max) / 2

                if width < size_threshold_width or height < size_threshold_height:
                    pbar.update(1)
                    continue

                # Step 3: this object's hard 0/1 shape (edge at 0.5).
                obj = ((owner == comp) & (soft >= 0.5)).astype(np.float32)
                if not obj.any():
                    pbar.update(1)
                    continue

                if mode == "convex_polygons":
                    polygon = self.get_mask_polygon(
                        obj.astype(np.uint8), max_poly_points
                    )
                    if polygon is None or len(polygon) < 3:
                        pbar.update(1)
                        continue
                    shape = self.polygon_to_mask(polygon, (H, W), supersample)
                elif mode == "box":
                    oy = np.where(np.any(obj, axis=1))[0]
                    ox = np.where(np.any(obj, axis=0))[0]
                    shape = np.zeros((H, W), dtype=np.float32)
                    shape[oy[0] : oy[-1] + 1, ox[0] : ox[-1] + 1] = 1.0
                else:  # "area"
                    shape = obj

                # Step 4-7: edge refinement.
                out = self.refine(shape, edge_shift, smooth, feather, contrast)
                if out.max() > 0:
                    separated.append(
                        (
                            centroid_x,
                            torch.from_numpy(
                                np.ascontiguousarray(out, dtype=np.float32)
                            ),
                        )
                    )
                pbar.update(1)

        if len(separated) > 0:
            separated.sort(key=lambda x: x[0])
            out_masks = torch.stack([m for _, m in separated], dim=0).to(mask.device)
            return (out_masks,)

        return (torch.zeros((1, H, W), dtype=torch.float32, device=mask.device),)

    """
        The node will always be re executed if any of the inputs change. Uncomment this to
        force it to run again even when the inputs don't change. Return a value that differs
        between runs to trigger re-execution.
    """
    # @classmethod
    # def IS_CHANGED(s, mask, invert_mask, size_threshold_width, size_threshold_height, mode, max_poly_points, **kwargs):
    #     return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AHP_SeparateMasks": AHP_SeparateMasks,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AHP_SeparateMasks": "AHP Separate Masks",
}
