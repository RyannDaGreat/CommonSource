import rp
import cv2
import numpy as np
from PIL import Image

class BackgroundRemover(rp.CachedInstances):
    """
    A class that removes the background from images and returns RGBA images with transparency.

    Example usage:
        background_remover = TransparentBackgroundRemover('cuda')
        input_path = 'https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/440px-Lenna_%28test_image%29.png'
        input_image = rp.load_image(input_path)
        rgba_image = background_remover(input_image)
        rp.display_alpha_image(rgba_image)
    """

    def __init__(self, device, checkpoint_path='transparent-background/ckpt_base.pth'):
        try:
            from transparent_background import Remover
        except ImportError:
            print("The 'transparent-background' package is not installed. Attempting to install it now...")
            pip_import('transparent_background', 'transparent-background', auto_yes=True)
            from transparent_background import Remover

        self.checkpoint_path = checkpoint_path

        if not rp.file_exists(checkpoint_path):
            checkpoint_folder = rp.get_path_parent(checkpoint_path)
            checkpoint_url = 'https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth'
            print(f"Downloading checkpoint file from: {checkpoint_url}")
            print(f"Saving checkpoint file to: {checkpoint_path}")
            rp.make_directory(checkpoint_folder)
            rp.download_url(checkpoint_url, checkpoint_path)
            print("Checkpoint file downloaded successfully.")

        self.remover = Remover(
            jit=True, 
            device=device, 
            ckpt=checkpoint_path
        )

    def __call__(self, image):
        """Returns an RGBA image"""
        image = rp.as_pil_image(rp.as_float_image(rp.as_rgb_image(image)))
        output = self.remover.process(image, type='rgba')
        return output