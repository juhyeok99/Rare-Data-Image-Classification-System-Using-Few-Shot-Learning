"""
data/preprocessing.py

PSNR and SSIM quality metrics for filtering generated images.
Only images that clear both thresholds get kept for the dataset.
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim
from PIL import Image
import config


def _to_array(img):
    """Accept PIL Image or numpy array, always return float64 [0,1] array."""
    if isinstance(img, Image.Image):
        return np.array(img).astype(np.float64) / 255.0
    return img.astype(np.float64) / 255.0 if img.max() > 1.0 else img.astype(np.float64)


def psnr(original, generated):
    """
    Peak Signal-to-Noise Ratio between two images.
    Higher is better; anything below ~20 dB is typically pretty noisy.

    Returns float (dB), or inf if images are identical.
    """
    orig = _to_array(original)
    gen  = _to_array(generated)

    mse = np.mean((orig - gen) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def ssim_score(original, generated):
    """
    Structural Similarity Index between two images.
    Range [-1, 1], higher means more similar structure.
    """
    orig = _to_array(original)
    gen  = _to_array(generated)

    # skimage expects channel_axis for RGB
    if orig.ndim == 3:
        score = _ssim(orig, gen, channel_axis=-1, data_range=1.0)
    else:
        score = _ssim(orig, gen, data_range=1.0)

    return float(score)


def quality_filter(original_img, generated_img,
                   psnr_thresh=config.PSNR_THRESHOLD,
                   ssim_thresh=config.SSIM_THRESHOLD):
    """
    Returns True if the generated image passes both quality gates.
    Images that are too noisy or structurally far from the source get dropped.
    """
    p = psnr(original_img, generated_img)
    s = ssim_score(original_img, generated_img)
    return (p >= psnr_thresh) and (s >= ssim_thresh)


def scale_image(img, size=(config.IMG_SIZE, config.IMG_SIZE)):
    """Resize a PIL Image to the target size."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0
                              else img.astype(np.uint8))
    return img.resize(size, Image.BICUBIC)
