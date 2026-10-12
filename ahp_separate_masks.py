import cv2
import numpy as np
import scipy.ndimage
import torch
from comfy.utils import ProgressBar


class AHP_SeparateMasks:
    """
    Splits a mask into one mask per connected component.

    - "area":            keeps each component's original soft (anti-aliased) values
    - "convex_polygons": anti-aliased convex hull with a limited number of points
    - "box":             filled bounding box (hard edges unless `feather` is set)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "size_threshold_width": (
                    "INT",
                    {"default": 256, "min": 0, "max": 4096, "step": 1},
                ),
                "size_threshold_height": (
                    "INT",
                    {"default": 256, "min": 0, "max": 4096, "step": 1},
                ),
                "mode": (["convex_polygons", "area", "box"],),
                "max_poly_points": (
                    "INT",
                    {"default": 8, "min": 3, "max": 32, "step": 1},
                ),
            },
            "optional": {
                # Pixels at or above this value are used to find components.
                "detect_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                # Faint pixels within this many pixels of a component are kept with it (area mode).
                "edge_reach": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                # Gaussian blur sigma applied to the output. 0 = off.
                "feather": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.5},
                ),
                # Anti-aliasing quality for polygon mode (1 = off).
                "supersample": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "separate"
    CATEGORY = "AHP_Tools/masking"
    OUTPUT_NODE = True
    DESCRIPTION = "Separates a mask into multiple masks based on the size of the connected components."

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

    def separate(
        self,
        mask: torch.Tensor,
        size_threshold_width: int,
        size_threshold_height: int,
        max_poly_points: int,
        mode: str,
        detect_threshold: float = 0.5,
        edge_reach: int = 2,
        feather: float = 0.0,
        supersample: int = 4,
    ):
        B, H, W = mask.shape
        soft_all = (
            mask.detach().float().clamp(0, 1).cpu().numpy()
        )  # original soft values, NOT rounded
        separated = []

        for b in range(B):
            soft = soft_all[b]

            # The binary copy is used ONLY to find connected components.
            binary = (soft >= detect_threshold).astype(np.uint8)
            labeled, ncomponents = scipy.ndimage.label(
                binary, structure=np.ones((3, 3), dtype=np.uint8)
            )
            if ncomponents == 0:
                continue

            # Give each faint pixel the label of its nearest component, but only if it is
            # within `edge_reach` pixels. This keeps anti-aliased edges with their object
            # without pulling distant haze into it.
            if (labeled == 0).any():
                dist, (iy, ix) = scipy.ndimage.distance_transform_edt(
                    labeled == 0, return_indices=True
                )
                owner = labeled[iy, ix]
                owner[(dist > edge_reach) | (soft <= 0)] = 0
            else:
                owner = labeled.copy()

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

                if width >= size_threshold_width and height >= size_threshold_height:
                    out = None
                    if mode == "convex_polygons":
                        polygon = self.get_mask_polygon(
                            comp_bin.astype(np.uint8), max_poly_points
                        )
                        if polygon is not None and len(polygon) >= 3:
                            out = self.polygon_to_mask(polygon, (H, W), supersample)
                    elif mode == "box":
                        out = np.zeros((H, W), dtype=np.float32)
                        out[y_min : y_max + 1, x_min : x_max + 1] = 1.0
                    else:  # "area"
                        out = np.where(owner == comp, soft, 0.0).astype(np.float32)

                    if out is not None:
                        if feather > 0:
                            out = cv2.GaussianBlur(out, (0, 0), sigmaX=feather)
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
