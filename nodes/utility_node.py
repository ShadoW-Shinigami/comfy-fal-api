import math
import numpy as np
from PIL import Image


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


class AspectRatioFinder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "width": ("NUMBER",),
                "height": ("NUMBER",),
                "aspect_ratio_mode": (["preset", "custom"],),
                "custom_aspect_ratios": ("STRING", {"default": "9:16, 16:9, 1:1, 4:3, 3:4", "multiline": False}),
            }
        }

    RETURN_TYPES = ("NUMBER", "FLOAT", "NUMBER", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("aspect_number", "aspect_float", "is_landscape_bool", "aspect_ratio_common", "aspect_type", "closest_aspect_ratio")
    FUNCTION = "aspect"

    CATEGORY = "FAL/Utils"

    def find_closest_aspect_ratio(self, target_ratio, aspect_ratios_str):
        """Find the closest aspect ratio from a comma-separated string of ratios."""
        # Default preset ratios
        preset_ratios = ["9:16", "16:9", "1:1", "4:3", "3:4"]

        # Parse the aspect ratios string
        if aspect_ratios_str and aspect_ratios_str.strip():
            ratio_strings = [r.strip() for r in aspect_ratios_str.split(",") if r.strip()]
        else:
            ratio_strings = preset_ratios

        closest_ratio = None
        min_diff = float('inf')

        for ratio_str in ratio_strings:
            try:
                if ':' in ratio_str:
                    w, h = ratio_str.split(':')
                    ratio_value = float(w.strip()) / float(h.strip())
                else:
                    ratio_value = float(ratio_str)

                diff = abs(target_ratio - ratio_value)
                if diff < min_diff:
                    min_diff = diff
                    closest_ratio = ratio_str.strip()
            except (ValueError, ZeroDivisionError):
                continue

        return closest_ratio if closest_ratio else "1:1"

    def aspect(self, boolean=True, image=None, width=None, height=None, aspect_ratio_mode="preset", custom_aspect_ratios="9:16, 16:9, 1:1, 4:3, 3:4"):

        if width and height:
            width = width; height = height
        elif image is not None:
            width, height = tensor2pil(image).size
        else:
            raise Exception("AspectRatioFinder must have width and height provided if no image tensor supplied.")

        aspect_ratio = width / height
        aspect_type = "landscape" if aspect_ratio > 1 else "portrait" if aspect_ratio < 1 else "square"

        landscape_bool = 0
        if aspect_type == "landscape":
            landscape_bool = 1

        gcd = math.gcd(width, height)
        gcd_w = width // gcd
        gcd_h = height // gcd
        aspect_ratio_common = f"{gcd_w}:{gcd_h}"

        # Find closest aspect ratio
        if aspect_ratio_mode == "custom" and custom_aspect_ratios:
            closest_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, custom_aspect_ratios)
        else:
            closest_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, "9:16, 16:9, 1:1, 4:3, 3:4")

        return aspect_ratio, aspect_ratio, landscape_bool, aspect_ratio_common, aspect_type, closest_aspect_ratio


NODE_CLASS_MAPPINGS = {
    "AspectRatioFinder": AspectRatioFinder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioFinder": "Aspect Ratio Finder",
}
