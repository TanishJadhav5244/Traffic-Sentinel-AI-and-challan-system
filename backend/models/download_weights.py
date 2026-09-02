import os
import sys
import urllib.request
import urllib.error

# ──────────────────────────────────────────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
PLATE_WEIGHTS_PATH = os.path.join(MODELS_DIR, "plate_detector.pt")
HELMET_WEIGHTS_PATH = os.path.join(MODELS_DIR, "helmet_detector.pt")

# Combined model path (checked first — used if you supply your own trained weights)
COMBINED_WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_detector.pt")

# ──────────────────────────────────────────────────────────────────────────────
# Public model URLs (no auth required)
#
# License Plate detector — keremberke/license-plate-detection
# Mirrored via GitHub Releases (avoids HuggingFace login requirement)
# yolov8n fine-tuned on CCPD + OpenImages, class: {0: 'license-plate'}
# ──────────────────────────────────────────────────────────────────────────────
PLATE_MODEL_URL = (
    "https://github.com/keremberke/awesome-yolov8-models/releases/download/v1.0.0/license-plate-detection-best.pt"
)

# Helmet detector
# yolov8n fine-tuned on helmet dataset, classes include helmet / no-helmet / person
HELMET_MODEL_URL = (
    "https://github.com/keremberke/awesome-yolov8-models/releases/download/v1.0.0/helmet-detection-best.pt"
)

# Fallback: if both specialty models fail, use generic yolov8n (COCO)
FALLBACK_YOLO_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
)


def _download_file(url: str, dest_path: str, label: str = "Model") -> bool:
    """Downloads a file from a URL and saves it locally. Returns True on success."""
    print(f"[Model Loader] Downloading {label} from:\n  {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "TrafficSentinelAI/2.0",
                "Accept": "application/octet-stream",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as response, \
                open(dest_path, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB chunks
            while True:
                block = response.read(chunk)
                if not block:
                    break
                out_file.write(block)
                downloaded += len(block)
                if total > 0:
                    pct = downloaded / total * 100
                    sys.stdout.write(f"\r  Progress: {pct:.1f}%  ({downloaded // 1024} KB)")
                    sys.stdout.flush()
        print(f"\n[Model Loader] {label} saved to {dest_path}")
        return True
    except Exception as e:
        msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"\n[Model Loader] Download failed for {label}: {msg}")
        if os.path.exists(dest_path):
            os.remove(dest_path)  # clean up partial download
        return False


def _is_valid_model(path: str, min_bytes: int = 100_000) -> bool:
    """Returns True if the model file exists and is larger than the minimum size threshold."""
    return os.path.exists(path) and os.path.getsize(path) > min_bytes


def download_all_models(force: bool = False) -> dict:
    """
    Downloads the license-plate detector and helmet detector models.

    Returns a dict with paths:
        {
            "plate_detector":  str | None,
            "helmet_detector": str | None,
            "combined":        str | None,   # Only set if user supplied best_detector.pt
        }
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    result = {"plate_detector": None, "helmet_detector": None, "combined": None}

    # ── Combined model (user-supplied or previously trained) ────────────────
    if _is_valid_model(COMBINED_WEIGHTS_PATH):
        print(f"[Model Loader] Combined custom weights present: {COMBINED_WEIGHTS_PATH}")
        result["combined"] = COMBINED_WEIGHTS_PATH
        return result  # nothing else needed — detector.py will use this

    # ── License-plate detector ───────────────────────────────────────────────
    if _is_valid_model(PLATE_WEIGHTS_PATH) and not force:
        print(f"[Model Loader] Plate detector present: {PLATE_WEIGHTS_PATH}")
        result["plate_detector"] = PLATE_WEIGHTS_PATH
    else:
        ok = _download_file(PLATE_MODEL_URL, PLATE_WEIGHTS_PATH, "License-Plate Detector (yolov8n)")
        if ok and _is_valid_model(PLATE_WEIGHTS_PATH):
            result["plate_detector"] = PLATE_WEIGHTS_PATH

    # ── Helmet/Rider detector ────────────────────────────────────────────────
    if _is_valid_model(HELMET_WEIGHTS_PATH) and not force:
        print(f"[Model Loader] Helmet detector present: {HELMET_WEIGHTS_PATH}")
        result["helmet_detector"] = HELMET_WEIGHTS_PATH
    else:
        ok = _download_file(HELMET_MODEL_URL, HELMET_WEIGHTS_PATH, "Helmet Detector (yolov8n)")
        if ok and _is_valid_model(HELMET_WEIGHTS_PATH):
            result["helmet_detector"] = HELMET_WEIGHTS_PATH

    # ── Fallback: download generic yolov8n if specialty models fail ──────────
    if not result["plate_detector"] and not result["helmet_detector"]:
        fallback_path = os.path.join(MODELS_DIR, "yolov8n_coco.pt")
        if not _is_valid_model(fallback_path):
            _download_file(FALLBACK_YOLO_URL, fallback_path, "YOLOv8n COCO (fallback)")
        print(
            "[Model Loader] WARNING: Only COCO fallback is available.\n"
            "  Helmet and license plate detection will use heuristic/contour methods.\n"
            "  For best results, place a trained best_detector.pt in the models/ folder."
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Legacy compat shim — called from the old "Generate Sample Assets" button
# ──────────────────────────────────────────────────────────────────────────────
def download_weights_if_missing(
    model_path: str = COMBINED_WEIGHTS_PATH,
    url: str = PLATE_MODEL_URL,
) -> str:
    """
    Backwards-compatible entry point.  Downloads all specialist models.
    Returns the path to the best available model.
    """
    result = download_all_models()
    # Return whatever is available, priority: combined > helmet > plate
    return (
        result.get("combined")
        or result.get("helmet_detector")
        or result.get("plate_detector")
        or "yolov8n.pt"
    )


if __name__ == "__main__":
    paths = download_all_models(force="--force" in sys.argv)
    print("\n[Model Loader] Summary:")
    for k, v in paths.items():
        print(f"  {k:20s}: {v or 'NOT AVAILABLE'}")

