from inspect import cleandoc

import cv2
import numpy as np
import scipy.ndimage
import torch
from comfy.utils import ProgressBar


class AHP_SeparateMasks:
    """
    Separates a mask into one mask per connected object, keeping the original soft (anti-aliased) edges.

    detect_threshold and bridge_gap only decide how pixels are grouped into objects.
    They never change the edge values of the output.

    Modes
    -----
    area:
        Each object keeps its original mask values, including anti-aliased edges.
        Together the outputs reproduce the input mask.
    convex_polygons:
        Anti-aliased convex hull with about `max_poly_points` vertices.
    box:
        Filled bounding box. Edges are hard unless `feather` is above 0.

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
                "mask": ("MASK", {"tooltip": "The mask to split into one mask per object."}),
                "size_threshold_width": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Ignore objects narrower than this many pixels.",
                }),
                "size_threshold_height": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Ignore objects shorter than this many pixels.",
                }),
                "mode": (["convex_polygons", "area", "box"], {
                    "tooltip": "area keeps the original edges. convex_polygons gives a simplified outline. box gives a bounding box.",
                }),
                "max_poly_points": ("INT", {
                    "default": 8,
                    "min": 3,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Target number of vertices in convex_polygons mode.",
                }),
            },
            "optional": {
                "detect_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                    "tooltip": "Pixels at or above this value are used to find objects. Only changes how objects are split. Output edges are not affected.",
                }),
                "bridge_gap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Joins pieces of one object separated by gaps up to about this many pixels. Use it if one object splits into many parts. 0 = off.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 16.0,
                    "step": 0.5,
                    "round": 0.001,
                    "display": "number",
                    "tooltip": "Blur radius applied to each output. 0 turns it off.",
                }),
                "supersample": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Anti-aliasing quality for convex_polygons mode. 1 turns it off.",
                }),
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
        """Anti-aliased rasterization: draw at higher resolution, then downsample with area averaging."""
        H, W = shape
        s = max(1, int(supersample))
        pts = np.round((polygon.astype(np.float64) + 0.5) * s - 0.5).astype(np.int32)
        big = np.zeros((H * s, W * s), dtype=np.uint8)
        cv2.fillPoly(big, [pts], 255)
        small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

    def get_mask_polygon(self, mask_np, max_points):
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

    def separate(self, mask: torch.Tensor, size_threshold_width: int, size_threshold_height: int,
                 max_poly_points: int, mode: str, detect_threshold: float = 0.5,
                 bridge_gap: int = 0, feather: float = 0.0, supersample: int = 4):
        B, H, W = mask.shape
        soft_all = mask.detach().float().clamp(0, 1).cpu().numpy()  # original soft values, NOT rounded
        structure = np.ones((3, 3), dtype=np.uint8)
        # The edge level used for polygon and box shapes. Independent of detect_threshold.
        level = min(0.5, detect_threshold)
        separated = []

        for b in range(B):
            soft = soft_all[b]

            # Step 1: decide which pixels form objects (this is the only place the threshold is used).
            binary = soft >= detect_threshold
            if bridge_gap > 0:
                binary = scipy.ndimage.binary_closing(binary, structure=structure, iterations=int(bridge_gap))
            labeled, ncomponents = scipy.ndimage.label(binary.astype(np.uint8), structure=structure)
            if ncomponents == 0:
                continue

            # Step 2: every pixel that has a mask value belongs to its nearest object.
            # The outputs therefore always carry the original values, whatever the threshold.
            if (labeled == 0).any():
                _, (iy, ix) = scipy.ndimage.distance_transform_edt(labeled == 0, return_indices=True)
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

                region = owner == comp
                core = region & (soft >= level)
                if not core.any():
                    pbar.update(1)
                    continue

                out = None
                if mode == "convex_polygons":
                    polygon = self.get_mask_polygon(core.astype(np.uint8), max_poly_points)
                    if polygon is not None and len(polygon) >= 3:
                        out = self.polygon_to_mask(polygon, (H, W), supersample)
                elif mode == "box":
                    cy = np.where(np.any(core, axis=1))[0]
                    cx = np.where(np.any(core, axis=0))[0]
                    out = np.zeros((H, W), dtype=np.float32)
                    out[cy[0]:cy[-1] + 1, cx[0]:cx[-1] + 1] = 1.0
                else:  # "area"
                    out = np.where(region, soft, 0.0).astype(np.float32)

                if out is not None:
                    if feather > 0:
                        out = cv2.GaussianBlur(out, (0, 0), sigmaX=feather)
                    separated.append((centroid_x, torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))))
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
    # def IS_CHANGED(s, mask, size_threshold_width, size_threshold_height, mode, max_poly_points, **kwargs):
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
