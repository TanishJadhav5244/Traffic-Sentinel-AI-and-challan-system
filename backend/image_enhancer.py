"""
backend/image_enhancer.py
=========================
Night-mode / low-light image enhancement pipeline for license plate crops.

Stages (applied selectively based on scene luminance):
  1. Auto low-light detection  -- measures mean luminance of the crop
  2. White balance correction  -- Gray World assumption to remove colour casts
  3. Adaptive gamma correction -- raises brightness for dark images
  4. CLAHE                     -- boosts local contrast while suppressing noise
  5. Multi-Scale Retinex (MSR) -- removes uneven lighting / shadows
  6. Fast bilateral denoising  -- smooths sensor noise while keeping character edges
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
_LOW_LIGHT_THRESHOLD = 85
_VERY_DARK_THRESHOLD = 40
_CLAHE_CLIP_LIMIT    = 3.0
_CLAHE_TILE_GRID     = (8, 8)
_MSR_SIGMAS          = (15, 80, 250)
_DENOISE_H           = 10
_DENOISE_TEMPLATE_WS = 7
_DENOISE_SEARCH_WS   = 21


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_low_light(img, threshold=_LOW_LIGHT_THRESHOLD):
    """Returns True if mean luminance is below threshold."""
    if img is None or img.size == 0:
        return False
    gray = _to_gray(img)
    return float(np.mean(gray)) < threshold


def get_luminance(img):
    """Returns mean luminance (0-255) of an image."""
    if img is None or img.size == 0:
        return 0.0
    return float(np.mean(_to_gray(img)))


def enhance(img, force=False):
    """
    Main entry-point. Runs the full enhancement pipeline on a BGR plate crop.

    Args:
        img   (np.ndarray): Input BGR image.
        force (bool):       Apply enhancement even if the image is well-lit.

    Returns:
        dict with keys:
            'enhanced'         : np.ndarray  -- enhanced BGR image
            'was_low_light'    : bool
            'luminance_before' : float
            'luminance_after'  : float
            'stages_applied'   : list[str]
    """
    if img is None or img.size == 0:
        return _null_result(img)

    lum_before = get_luminance(img)
    low_light  = lum_before < _LOW_LIGHT_THRESHOLD
    stages     = []

    if not (low_light or force):
        return {
            "enhanced"         : img,
            "was_low_light"    : False,
            "luminance_before" : lum_before,
            "luminance_after"  : lum_before,
            "stages_applied"   : [],
        }

    out = img.copy()

    # Stage 1 -- White balance
    out = _white_balance(out)
    stages.append("white_balance")

    # Stage 2 -- Adaptive gamma correction
    gamma = _compute_gamma(lum_before)
    out = _apply_gamma(out, gamma)
    stages.append(f"gamma({gamma:.2f})")

    # Stage 3 -- CLAHE on luminance channel
    out = _apply_clahe(out)
    stages.append("CLAHE")

    # Stage 4 -- Multi-Scale Retinex (for very dark or uneven images)
    if lum_before < _VERY_DARK_THRESHOLD or _has_strong_shadows(img):
        out = _multi_scale_retinex(out)
        stages.append("MSR")

    # Stage 5 -- Denoising
    out = _denoise(out)
    stages.append("denoise")

    lum_after = get_luminance(out)

    return {
        "enhanced"         : out,
        "was_low_light"    : True,
        "luminance_before" : lum_before,
        "luminance_after"  : lum_after,
        "stages_applied"   : stages,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _null_result(img):
    return {
        "enhanced"         : img,
        "was_low_light"    : False,
        "luminance_before" : 0.0,
        "luminance_after"  : 0.0,
        "stages_applied"   : [],
    }


def _to_gray(img):
    if img is None:
        return np.array([])
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _white_balance(img):
    """Gray World white balance -- removes artificial lamp colour casts."""
    result = img.copy().astype(np.float32)
    b, g, r = cv2.split(result)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    overall_mean = (b_mean + g_mean + r_mean) / 3.0
    if b_mean > 0:
        b = np.clip(b * (overall_mean / b_mean), 0, 255)
    if g_mean > 0:
        g = np.clip(g * (overall_mean / g_mean), 0, 255)
    if r_mean > 0:
        r = np.clip(r * (overall_mean / r_mean), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def _compute_gamma(mean_lum):
    """Adaptive gamma: darker images get a lower gamma (bigger brightness boost)."""
    if mean_lum < 20:
        return 0.35
    elif mean_lum < 40:
        return 0.45
    elif mean_lum < 60:
        return 0.55
    elif mean_lum < 85:
        return 0.70
    return 1.0


def _apply_gamma(img, gamma):
    """LUT-based gamma correction (very fast)."""
    if gamma == 1.0:
        return img
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(img, table)


def _apply_clahe(img):
    """CLAHE on the L-channel of LAB colourspace to boost local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT,
                             tileGridSize=_CLAHE_TILE_GRID)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _single_scale_retinex(img_f, sigma):
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
    blurred = np.where(blurred < 1.0, 1.0, blurred)
    return np.log1p(img_f) - np.log1p(blurred)


def _multi_scale_retinex(img, sigmas=_MSR_SIGMAS,
                          low_clip=0.01, high_clip=0.99):
    """Multi-Scale Retinex -- equalises illumination under uneven street lighting."""
    img_f = img.astype(np.float32) + 1.0
    msr = np.zeros_like(img_f)
    for sigma in sigmas:
        msr += _single_scale_retinex(img_f, sigma)
    msr /= len(sigmas)

    result = np.zeros_like(img_f)
    for c in range(img_f.shape[2]):
        lo = np.percentile(msr[:, :, c], low_clip  * 100)
        hi = np.percentile(msr[:, :, c], high_clip * 100)
        if hi - lo < 1e-3:
            result[:, :, c] = img_f[:, :, c] - 1.0
        else:
            ch = np.clip(msr[:, :, c], lo, hi)
            ch = (ch - lo) / (hi - lo + 1e-6) * 255.0
            result[:, :, c] = ch
    return result.astype(np.uint8)


def _denoise(img):
    """Non-Local Means denoising to suppress noise amplified by earlier stages."""
    if len(img.shape) == 2:
        return cv2.fastNlMeansDenoising(
            img, h=_DENOISE_H,
            templateWindowSize=_DENOISE_TEMPLATE_WS,
            searchWindowSize=_DENOISE_SEARCH_WS
        )
    return cv2.fastNlMeansDenoisingColored(
        img,
        h=_DENOISE_H, hColor=_DENOISE_H,
        templateWindowSize=_DENOISE_TEMPLATE_WS,
        searchWindowSize=_DENOISE_SEARCH_WS
    )


def _has_strong_shadows(img, shadow_fraction=0.40):
    """True if >40% of pixels are significantly darker than the mean."""
    gray = _to_gray(img).astype(np.float32)
    mean_lum = np.mean(gray)
    dark_fraction = np.mean(gray < mean_lum * 0.5)
    return bool(dark_fraction > shadow_fraction)
