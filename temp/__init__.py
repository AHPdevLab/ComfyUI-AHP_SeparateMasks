from .ah_separate_masks import AH_SeparateMasks

NODE_CLASS_MAPPINGS = {
    "AH_SeparateMasks": AH_SeparateMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AH_SeparateMasks": "AH Separate Masks",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
