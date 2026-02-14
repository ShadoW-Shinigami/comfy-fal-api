import configparser
import io
import math
import os
import tempfile

import numpy as np
import requests
import torch
from fal_client.client import SyncClient
from PIL import Image


class FalConfig:
    """Singleton class to handle FAL configuration and client setup."""

    _instance = None
    _client = None
    _key = None
    _key_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if os.environ.get("FAL_KEY") is not None:
                print("FAL_KEY found in environment variables")
                self._key = os.environ["FAL_KEY"]
            else:
                print("FAL_KEY not found in environment variables")
                self._key = config["API"]["FAL_KEY"]
                print("FAL_KEY found in config.ini")
                os.environ["FAL_KEY"] = self._key
                print("FAL_KEY set in environment variables")

            # Check if FAL key is the default placeholder
            if self._key == "<your_fal_api_key_here>":
                print("WARNING: You are using the default FAL API key placeholder!")
                print("Please set your actual FAL API key in either:")
                print("1. The config.ini file under [API] section")
                print("2. Or as an environment variable named FAL_KEY")
                print("Get your API key from: https://fal.ai/dashboard/keys")
        except KeyError:
            print("Error: FAL_KEY not found in config.ini or environment variables")

    def get_client(self):
        """Get or create the FAL client."""
        if self._client is None:
            self._client = SyncClient(key=self._key)
        return self._client

    def set_key(self, key, name=None):
        """Set a new API key at runtime, resetting the client."""
        self._key = key
        self._key_name = name
        self._client = None
        os.environ["FAL_KEY"] = key
        print(f"FAL API key switched to: {name or 'unnamed'}")

    def get_key_name(self):
        """Get the display name of the active key."""
        if self._key_name:
            return self._key_name
        return "config.ini / env"

    def get_key(self):
        """Get the FAL API key."""
        return self._key


class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image):
        """Upload image tensor to FAL and return URL."""
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            # Upload the temporary file
            client = FalConfig().get_client()
            image_url = client.upload_file(temp_file_path)
            return image_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)
                
    @staticmethod
    def upload_file(file_path):
        """Upload a file to FAL and return URL."""
        try:
            client = FalConfig().get_client()
            file_url = client.upload_file(file_path)
            return file_url
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None
        
    @staticmethod
    def mask_to_image(mask):
        """Convert mask tensor to image tensor."""
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result
    
    @staticmethod
    def prepare_images(images):
        """Preprocess images for use with FAL."""
        image_urls = []
        if images is not None:

                if isinstance(images, torch.Tensor):
                    if images.ndim == 4 and images.shape[0] > 1:
                        for i in range(images.shape[0]):
                            single_img = images[i:i+1]
                            img_url = ImageUtils.upload_image(single_img)
                            if img_url:
                                image_urls.append(img_url)
                    else:
                        img_url = ImageUtils.upload_image(images)
                        if img_url:
                            image_urls.append(img_url)

                elif isinstance(images, (list, tuple)):
                    for img in images:
                        img_url = ImageUtils.upload_image(img)
                        if img_url:
                            image_urls.append(img_url)
        return image_urls


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing single image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class ApiHandler:
    """Utility functions for API interactions."""

    @staticmethod
    def submit_and_get_result(endpoint, arguments):
        """Submit job to FAL API and get result."""
        try:
            client = FalConfig().get_client()
            handler = client.submit(endpoint, arguments=arguments)
            return handler.get()
        except Exception as e:
            print(f"Error submitting to {endpoint}: {str(e)}")
            raise e

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        print(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)


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
