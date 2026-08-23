import os
import re
import cv2
import numpy as np

from backend.image_enhancer import enhance as enhance_image, is_low_light

# Lazy load OCR engines to prevent startup delays if one isn't used
_easyocr_reader = None
_tesseract_available = None

class LicensePlateOCR:
    def __init__(self, config=None):
        """
        Initializes the OCR engine wrapper.
        
        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config or {}
        
        # Load OCR config
        self.ocr_config = self.config.get("ocr", {})
        self.default_engine = self.ocr_config.get("default_engine", "easyocr")
        
        self.preprocess_settings = self.config.get("ocr", {}).get("preprocessing", {
            "resize_factor": 2.0,
            "bilateral_filter": True,
            "adaptive_threshold": True,
            "deskew": True
        })

        self.enhancer_settings = self.config.get("ocr", {}).get("enhancer", {
            "auto_low_light": True,
            "luminance_threshold": 85,
            "force_enhancement": False
        })
        
        # Configure Tesseract path
        tesseract_path = self.ocr_config.get("tesseract_path", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        self._configure_tesseract(tesseract_path)

    @property
    def engine_type(self):
        return self.default_engine

    def _configure_tesseract(self, path):
        """Sets up pytesseract path and validates if it's available."""
        global _tesseract_available
        try:
            import pytesseract
            # Set path
            pytesseract.pytesseract.tesseract_cmd = path
            # Simple check if executable works
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False

    def _get_easyocr_reader(self):
        """Initializes EasyOCR reader lazily to conserve memory/time."""
        global _easyocr_reader
        if _easyocr_reader is None:
            import easyocr
            import torch
            gpu_available = torch.cuda.is_available()
            print(f"[OCR] Initializing EasyOCR Reader (GPU={gpu_available})...")
            # En forces English recognition
            _easyocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
        return _easyocr_reader

    def preprocess_image(self, img):
        """
        Applies a pipeline of filters to optimize license plate image for OCR.
        
        Steps:
        1. Grayscale
        2. Upscaling (resize) to improve small character visibility
        3. Bilateral filter to smooth noise while keeping edges sharp
        4. Deskewing to align slightly rotated plates
        5. Otsu's thresholding (binarization) to isolate characters
        """
        if img is None or img.size == 0:
            return None
            
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize / Upscale
        factor = self.preprocess_settings.get("resize_factor", 2.0)
        if factor != 1.0:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
            
        # 3. Noise removal (Bilateral Filter preserves edges better than Gaussian blur)
        if self.preprocess_settings.get("bilateral_filter", True):
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
        # 4. Deskewing (rotation correction)
        if self.preprocess_settings.get("deskew", True):
            gray = self.deskew(gray)
            
        # 5. Adaptive Thresholding / Binarization (Otsu's method)
        if self.preprocess_settings.get("adaptive_threshold", True):
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Ensure background is white and text is black (or vice versa depending on OCR)
            # Usually, OCR prefers black text on white background
            # If the edges of the image are mostly white, we assume white background.
            # If mostly black, we invert the image.
            edge_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
            if np.mean(edge_pixels) < 127:
                binary = cv2.bitwise_not(binary)
            return binary
            
        return gray

    def deskew(self, img):
        """Corrects small rotations/skews in the cropped license plate image."""
        try:
            # Find coordinates of all non-zero (white) pixels
            # Since deskew runs after binarization or on edge-detected image
            coords = np.column_stack(np.where(img > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle depending on skew direction
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Ignore tiny angles to prevent unnecessary warping
            if abs(angle) < 0.5 or abs(angle) > 20:
                return img
                
            # Perform rotation
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return img

    def clean_plate_text(self, text):
        """
        Cleans and standardizes OCR text using Indian license plate structure heuristics.
        Handles formats:
        - 10 chars: LL NN LL NNNN (e.g., MH12DE5678)
        - 9 chars:  LL NN L NNNN  (e.g., KA03M1234)
        - 8 chars:  LL N L NNNN   (e.g., DL3C1234)
        """
        if not text:
            return ""
            
        # Remove special characters, punctuation, and spaces
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        
        # 10-character Indian plate format (State:2, District:2, Series:2, Number:4)
        if len(cleaned) == 10:
            chars = list(cleaned)
            # Positions 0, 1: State code (letters)
            for i in [0, 1]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
                if chars[i] == '8': chars[i] = 'B'
            # Positions 2, 3: District code (digits)
            for i in [2, 3]:
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'Z': chars[i] = '2'
                if chars[i] == 'B': chars[i] = '8'
            # Positions 4, 5: Series letters
            for i in [4, 5]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
                if chars[i] == '8': chars[i] = 'B'
            # Positions 6, 7, 8, 9: Digits
            for i in range(6, 10):
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'Z': chars[i] = '2'
                if chars[i] == 'B': chars[i] = '8'
                if chars[i] == 'G': chars[i] = '6'
            cleaned = "".join(chars)
            
        # 9-character format (State:2, District:2, Series:1, Number:4)
        elif len(cleaned) == 9:
            chars = list(cleaned)
            for i in [0, 1]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
            for i in [2, 3]:
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
            if chars[4] == '0': chars[4] = 'O'
            if chars[4] == '1': chars[4] = 'I'
            for i in range(5, 9):
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'Z': chars[i] = '2'
            cleaned = "".join(chars)
            
        # 8-character format (State:2, District:1, Series:1, Number:4) e.g., DL3C1234
        elif len(cleaned) == 8:
            chars = list(cleaned)
            for i in [0, 1]:
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
            if chars[2] in ['O', 'I', 'S']:
                chars[2] = {'O': '0', 'I': '1', 'S': '5'}[chars[2]]
            if chars[3] in ['0', '1']:
                chars[3] = {'0': 'O', '1': 'I'}[chars[3]]
            for i in range(4, 8):
                if chars[i] in ['O', 'I', 'S', 'Z']:
                    chars[i] = {'O': '0', 'I': '1', 'S': '5', 'Z': '2'}[chars[i]]
            cleaned = "".join(chars)
            
        return cleaned

    def run_easyocr(self, img):
        """Runs EasyOCR on the image and returns (text, confidence)."""
        try:
            reader = self._get_easyocr_reader()
            results = reader.readtext(img)
            if not results:
                return "", 0.0
                
            # Combine all text blocks
            texts = []
            confidences = []
            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
                
            combined_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return combined_text, avg_conf
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
            return "", 0.0

    def run_tesseract(self, img):
        """Runs Tesseract OCR on the image and returns (text, confidence)."""
        global _tesseract_available
        if not _tesseract_available:
            return "", 0.0
            
        try:
            import pytesseract
            # We use image_to_data to get word confidences
            # config parameter to treat plate as a single line: psm 7 or psm 8
            custom_config = r'--oem 3 --psm 7'
            data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract words and confidences
            words = []
            confidences = []
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                conf = float(data['conf'][i])
                if word != "" and conf != -1:
                    words.append(word)
                    confidences.append(conf / 100.0) # Convert to 0-1 range
                    
            combined_text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return combined_text, avg_conf
        except Exception as e:
            print(f"[OCR] Tesseract error: {e}")
            return "", 0.0

    def recognize(self, img, engine=None, preprocess=True, enable_enhancer=True, force_enhancer=False):
        """
        Recognizes text from the image using the specified OCR engine.
        
        Args:
            img (numpy.ndarray): Cropped BGR plate image.
            engine (str): "easyocr" or "tesseract". If None, uses default_engine.
            preprocess (bool): Whether to apply image preprocessing filters.
            enable_enhancer (bool): Auto-run night-mode enhancement for low-light crops.
            force_enhancer (bool): Force enhancement even if luminance is normal.
            
        Returns:
            dict: {
                "raw_text": str,
                "cleaned_text": str,
                "confidence": float,
                "engine_used": str,
                "preprocessed_img": numpy.ndarray,
                "enhanced_img": numpy.ndarray,
                "was_low_light": bool,
                "enhancer_stages": list
            }
        """
        if img is None or img.size == 0:
            return {
                "raw_text": "", "cleaned_text": "", "confidence": 0.0,
                "engine_used": "none", "preprocessed_img": None,
                "enhanced_img": None, "was_low_light": False, "enhancer_stages": []
            }
            
        # Optional night-mode / low-light enhancement
        enhanced_img = img
        was_low_light = False
        enhancer_stages = []

        auto_enhance = self.enhancer_settings.get("auto_low_light", True)
        if enable_enhancer and (auto_enhance or force_enhancer):
            enh_res = enhance_image(img, force=force_enhancer)
            enhanced_img = enh_res.get("enhanced", img)
            was_low_light = enh_res.get("was_low_light", False)
            enhancer_stages = enh_res.get("stages_applied", [])

        # Apply preprocessing on enhanced image
        preprocessed = self.preprocess_image(enhanced_img) if preprocess else enhanced_img
        
        # Decide engine
        engine = engine or self.default_engine
        
        plate_regex = self.ocr_config.get("plate_regex", r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

        def evaluate_candidate(cand_img, eng):
            if eng == "tesseract" and _tesseract_available:
                raw, conf = self.run_tesseract(cand_img)
            else:
                raw, conf = self.run_easyocr(cand_img)
            clean = self.clean_plate_text(raw)
            # Regex match bonus (gives +1.0 boost if clean matches Indian plate pattern)
            is_valid_format = bool(re.match(plate_regex, clean))
            score = conf + (1.0 if is_valid_format else 0.0) + (0.3 if len(clean) in [8, 9, 10] else 0.0)
            return {
                "raw_text": raw,
                "cleaned_text": clean,
                "confidence": conf,
                "score": score,
                "is_valid": is_valid_format,
                "img": cand_img
            }

        # Multi-Pass OCR evaluation across preprocessed, contrast-enhanced, and grayscale variants
        candidates = []
        candidates.append(evaluate_candidate(preprocessed, engine))
        
        if preprocess:
            gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY) if len(enhanced_img.shape) == 3 else enhanced_img
            candidates.append(evaluate_candidate(gray_enhanced, engine))
            candidates.append(evaluate_candidate(enhanced_img, engine))

        # Pick best scoring candidate
        best_candidate = max(candidates, key=lambda c: c["score"])
        
        raw_text = best_candidate["raw_text"]
        cleaned_text = best_candidate["cleaned_text"]
        confidence = best_candidate["confidence"]
        
        # If tesseract wasn't available, actual engine used is easyocr
        actual_engine = "easyocr" if (engine == "tesseract" and not _tesseract_available) else engine

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "confidence": confidence,
            "engine_used": actual_engine,
            "preprocessed_img": best_candidate["img"],
            "enhanced_img": enhanced_img,
            "was_low_light": was_low_light,
            "enhancer_stages": enhancer_stages
        }
