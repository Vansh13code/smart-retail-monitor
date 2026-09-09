from typing import List, Tuple, Dict, Any

import base64
import io
import logging
import math
import cv2
import easyocr
import re
import numpy as np
import torch
import uuid
import os

from ultralytics import YOLO
from app.core.model_manager import model_manager
from app.core.ocr_cache import ocr_cache

from app.schemas.detection import Detection
from app.schemas.crop import Crop


class PriceTagDetector:
    """
    Detects price tags using best.pt model.
    """

    def __init__(self):

        logging.info("Loading Price Tag Detection Model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use centralized ModelManager to load best.pt
        self.model = model_manager.get_price_tag_model()

        logging.info("Price tag detection model loaded via centralized ModelManager")

    def detect(self, image, conf: float = 0.20, iou: float = 0.35) -> List[Detection]:
        """Find physical label ROIs; best.pt has no price-tag class.

        Running the product/shelf model here treated products as price tags.
        This contour detector identifies bright, short, horizontal shelf labels
        and is the only source of ROIs passed to EasyOCR.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = gray.shape[:2]
        # Shelf prices are printed on white/grey or yellow paper labels. Do
        # not use all bright pixels: that was grouping product packaging.
        white_label = cv2.inRange(hsv, (0, 0, 165), (180, 90, 255))
        yellow_label = cv2.inRange(hsv, (15, 75, 125), (42, 255, 255))
        mask = cv2.bitwise_or(white_label, yellow_label)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)
            # Labels are short, wide strips. Reject page borders and large
            # bright product regions before OCR is considered.
            if y < int(height * 0.12):
                continue
            if w < max(24, width // 30) or h < max(10, height // 55) or h > max(60, height // 9):
                continue
            if not 1.05 <= aspect <= 3.8 or w > width // 3 or area < 300 or area > width * height * 0.035:
                continue
            fill = cv2.contourArea(contour) / max(area, 1)
            confidence = min(0.99, 0.45 + 0.35 * min(1.0, aspect / 6) + 0.20 * fill)
            detections.append(Detection(
                id=len(detections), bbox=(x, y, x + w, y + h),
                confidence=float(confidence), class_id=-1, class_name="price_tag"
            ))
        return self._merge_overlapping(detections)

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _merge_overlapping(self, detections: List[Detection], iou_thresh: float = 0.6) -> List[Detection]:
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while detections:
            current = detections.pop(0)
            group = [current]
            remaining = []
            for d in detections:
                if self._iou(current.bbox, d.bbox) >= iou_thresh:
                    group.append(d)
                else:
                    remaining.append(d)
            detections = remaining

            # merge group into single detection
            if len(group) == 1:
                keep.append(current)
            else:
                xs = [g.bbox[0] for g in group] + [g.bbox[2] for g in group]
                ys = [g.bbox[1] for g in group] + [g.bbox[3] for g in group]
                merged_bbox = (min(xs), min(ys), max(xs), max(ys))
                avg_conf = float(sum(g.confidence for g in group) / len(group))
                keep.append(Detection(id=current.id, bbox=merged_bbox, confidence=avg_conf, class_id=current.class_id, class_name=current.class_name))

        # reassign ids
        for i, d in enumerate(keep):
            d.id = i
        return keep


class Cropper:

    def crop(
        self,
        image,
        detections: List[Detection]
    ) -> List[Crop]:
        crops = []

        for detection in detections:

            x1, y1, x2, y2 = detection.bbox
            
            # Ensure within image bounds
            h, w = image.shape[:2]
            x1_c = max(0, min(x1, w - 1))
            x2_c = max(0, min(x2, w))
            y1_c = max(0, min(y1, h - 1))
            y2_c = max(0, min(y2, h))

            roi = image[y1_c:y2_c, x1_c:x2_c]

            crops.append(
                Crop(
                    id=detection.id,
                    image=roi,
                    bbox=detection.bbox,
                    confidence=detection.confidence
                )
            )
        return crops


class ImagePreprocessor:
    """
    Enhanced preprocessing for cropped price tag images to improve OCR accuracy.
    Includes CLAHE, Adaptive Threshold, Gamma correction, Noise removal, Sharpening, Perspective correction.
    """

    def preprocess(self, crop: Crop) -> Any:
        img = crop.image.copy()

        # Check if empty crop
        if img.size == 0:
            return img

        # Upscale small crops to improve OCR on small text
        h, w = img.shape[:2]
        scale = 3 if max(h, w) < 150 else (2 if max(h, w) < 300 else 1)
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise with bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Additional median filter for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)

        # Sharpen with unsharp masking
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

        # Gamma correction to adjust brightness
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(sharpened, table)

        # Morphological operations to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        corrected = cv2.morphologyEx(corrected, cv2.MORPH_CLOSE, kernel)

        # Adaptive threshold for better text segmentation
        binary = cv2.adaptiveThreshold(
            corrected, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15, 8
        )

        # Deskew using moments / minAreaRect
        coords = np.column_stack(np.where(binary < 255))
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                (h_img, w_img) = corrected.shape[:2]
                center = (w_img // 2, h_img // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                corrected = cv2.warpAffine(
                    corrected, M, (w_img, h_img), 
                    flags=cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_REPLICATE
                )

        return corrected


class OCRService:
    """
    Extracts text from preprocessed price tag images
    using EasyOCR.
    """

    def __init__(self):

        logging.info("Loading EasyOCR Model...")

        use_gpu = torch.cuda.is_available()
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            logging.info("EasyOCR loaded (gpu=%s)", use_gpu)
        except Exception as e:
            logging.exception("Failed to initialize EasyOCR, falling back to cpu: %s", e)
            self.reader = easyocr.Reader(['en'], gpu=False)
        
        self.use_cache = True  # Enable OCR caching for performance

    def read(self, image, min_confidence: float = 0.30):
        # image is expected to be a binary or gray image
        if image.size == 0:
            return {"text": "", "confidence": 0.0, "blocks": []}

        # Check cache first if enabled
        if self.use_cache:
            cached_result = ocr_cache.get(image)
            if cached_result is not None:
                return cached_result

        # Run 1: Grayscale/Original preprocessed image
        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.3,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            width_ths=0.7,
            ycenter_ths=0.5,
            height_ths=0.5,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789₹RsMRP.-/:$ "
        )

        # If no results or low average confidence, try binary threshold image
        avg_conf = sum(conf for _, _, conf in results if conf is not None) / len(results) if results else 0.0
        if not results or avg_conf < 0.5:
            # Run 2: Adaptive threshold binary image
            bin_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            results_bin = self.reader.readtext(
                bin_image,
                detail=1,
                paragraph=False,
                text_threshold=0.5,
                low_text=0.3,
                link_threshold=0.3,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                width_ths=0.7,
                ycenter_ths=0.5,
                height_ths=0.5,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789₹RsMRP.-/:$ "
            )
            avg_conf_bin = sum(conf for _, _, conf in results_bin if conf is not None) / len(results_bin) if results_bin else 0.0
            if avg_conf_bin > avg_conf:
                results = results_bin

        # results -> list of (bbox, text, confidence)
        filtered = []
        for bbox, text, conf in results:
            if conf is None:
                continue
            if conf >= min_confidence and text.strip():
                filtered.append({"bbox": bbox, "text": text.strip(), "confidence": float(conf)})

        # Merge fragmented adjacent boxes
        merged = []
        used = [False] * len(filtered)
        for i, a in enumerate(filtered):
            if used[i]:
                continue
            group_text = [a["text"]]
            group_conf = [a["confidence"]]
            ax = np.array(a["bbox"]).reshape(-1, 2)
            aymin = ax[:, 1].min()
            aymax = ax[:, 1].max()
            used[i] = True
            for j, b in enumerate(filtered[i + 1 :], start=i + 1):
                if used[j]:
                    continue
                bx = np.array(b["bbox"]).reshape(-1, 2)
                bymin = bx[:, 1].min()
                bymax = bx[:, 1].max()
                overlap = max(0, min(aymax, bymax) - max(aymin, bymin))
                min_h = min(aymax - aymin, bymax - bymin)
                if min_h <= 0:
                    continue
                if overlap / min_h > 0.2:
                    group_text.append(b["text"])
                    group_conf.append(b["confidence"])
                    used[j] = True

            merged_text = " ".join(group_text)
            avg_conf = sum(group_conf) / len(group_conf) if group_conf else 0.0
            merged.append({"text": merged_text, "confidence": avg_conf})

        final_text = " ".join([m["text"] for m in merged])
        avg_confidence = sum([m["confidence"] for m in merged]) / len(merged) if merged else 0.0

        result = {"text": final_text, "confidence": float(avg_confidence), "blocks": merged}
        
        # Cache the result if caching is enabled
        if self.use_cache:
            ocr_cache.set(image, result)
        
        return result


class TextCleaner:
    """
    Cleans and normalizes OCR text before parsing.
    """

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Watermarks, URLs and long asset identifiers are never retail label
        # content.  Discard the complete OCR candidate, not just its price.
        if re.search(r"shutterstock|https?://|www\\.|\\b[a-z0-9_-]{10,}\\b", text, re.IGNORECASE):
            return ""

        text = text.upper()

        replacements = {
            "RS.": "RS ",
            "RS": "RS ",
            "MRP:": "MRP ",
            "PRICE:": "PRICE ",
            "SMARTPRICE": "SMART PRICE",
            "SMARTPRICE:": "SMART PRICE",
            "|": " ",
            "[": " ",
            "]": " ",
            "{": " ",
            "}": " ",
            "<": " ",
            ">": " ",
            "=": " ",
            "_": " ",
            "`": " ",
            "~": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Fix OCR issues where S/5 confusion happens
        text = re.sub(r'([0-9])S\b', r'\g<1>5', text)

        # Fix O/o confusion with 0 inside numbers
        text = re.sub(r'([0-9])[Oo]\b', r'\g<1>0', text)
        text = re.sub(r'\b[Oo]([0-9])', r'0\1', text)

        # Clean trailing slashes/dashes like "499/-"
        text = re.sub(r'/-\b', '', text)
        text = re.sub(r'/-', '', text)

        text = re.sub(
            r"[^A-Z0-9₹.\s\$]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )
        return text.strip()


class TextParser:
    """
    Parses OCR text and extracts useful information.
    """

    def parse(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        text = ocr_result.get("text", "")

        # Detect currency
        currency = ""
        for symbol in ["₹", "Rs", "RS", "INR", "$"]:
            if symbol.lower() in text.lower():
                currency = symbol
                break

        # Match standard prices like 150, 150.00, 1,500.00
        matches = re.findall(r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]{2})?", text)

        price = None
        if matches:
            candidates = []
            for m in matches:
                clean_m = m.replace(",", "")
                try:
                    val = float(clean_m)
                    # Skip typical long barcode numbers
                    if val >= 50000 or len(clean_m) >= 8:
                        continue
                    
                    # Score candidate
                    score = 0
                    
                    # Check context: check if price is near currency indicator or MRP/Price text
                    idx = text.find(m)
                    context_window = text[max(0, idx-8):min(len(text), idx+len(m)+8)].upper()
                    
                    if any(x in context_window for x in ["RS", "MRP", "₹", "$", "PRICE", "PRC"]):
                        score += 15
                    
                    # Decimal points (e.g. 49.00) are common in retail prices
                    if "." in m:
                        score += 5
                        
                    # Sensible price ranges (retail products usually 5 to 5000)
                    if 5.0 <= val <= 5000.0:
                        score += 5
                        
                    candidates.append((val, score))
                except ValueError:
                    continue
            
            if candidates:
                # Sort by score desc, then by value desc
                candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
                price = candidates[0][0]

        cleaned_text = text
        for symbol in ["₹", "Rs", "RS", "INR", "$"]:
            cleaned_text = re.sub(re.escape(symbol), "", cleaned_text, flags=re.IGNORECASE)
        # Remove price from text if found
        if price is not None:
            price_str = str(int(price))
            cleaned_text = cleaned_text.replace(price_str, "")

        cleaned_text = re.sub(r"[0-9]+(?:\.[0-9]+)?", "", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_text = re.sub(r"^[.\-/: ]+", "", cleaned_text)

        return {
            "product_name": cleaned_text if cleaned_text else "Retail Product",
            "price": price,
            "currency": currency if currency else "Rs",
            "raw_text": text
        }


class BusinessLogic:
    """
    Applies business rules to parsed OCR data.
    """

    def process(self, parsed_data):

        product_name = parsed_data.get("product_name", "").strip()
        price = parsed_data.get("price")

        result = {
            "product_name": product_name,
            "price": price,
            "status": "Valid"
        }

        if not product_name:
            result["status"] = "Product Name Missing"

        elif price is None:
            result["status"] = "Price Missing"

        return result


class PriceTagService:

    def __init__(self):
        self.detector = PriceTagDetector()
        self.cropper = Cropper()
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRService()
        self.cleaner = TextCleaner()
        self.parser = TextParser()
        self.business_logic = BusinessLogic()

    def process(self, frame):
        # Create uploads folder if missing
        os.makedirs(os.path.join("uploads", "crops"), exist_ok=True)

        detections = self.detector.detect(frame)

        if len(detections) == 0:
            # Never OCR the complete image: it reads watermarks/URLs (for
            # example Shutterstock) and produces fabricated product prices.
            # Return a visual result even when no price-tag ROI exists.
            annotated = frame.copy()
            cv2.putText(annotated, "No price tags detected", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            _, buffer = cv2.imencode('.png', annotated)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "total_price_tags": 0,
                "detections": [],
                "ocr_results": [],
                "cleaned_results": [],
                "parsed_results": [],
                "business_results": [],
                "annotated_image": f"data:image/png;base64,{annotated_b64}",
                "warnings": ["No price-tag boxes detected. OCR was not run on the full image."]
            }

        crops = self.cropper.crop(frame, detections)

        ocr_results = []
        cleaned_results = []
        parsed_results = []
        business_results = []
        detection_response = []
        processed_crops = [] 

        annotated = frame.copy()

        for detection, crop in zip(detections, crops):

            processed = self.preprocessor.preprocess(crop)

            processed_crops.append(processed)

            ocr = self.ocr.read(processed)

            cleaned = {
                "text": self.cleaner.clean(ocr.get("text", "")),
                "confidence": float(ocr.get("confidence", 0.0))
            }

            parsed = self.parser.parse(cleaned)

            business = self.business_logic.process(parsed)

            ocr_results.append(ocr)
            cleaned_results.append(cleaned)
            parsed_results.append(parsed)
            business_results.append(business)

            # Save the cropped image to disk
            crop_filename = f"crop_{uuid.uuid4().hex}.png"
            crop_filepath = os.path.join("uploads", "crops", crop_filename)
            cv2.imwrite(crop_filepath, crop.image)
            cropped_image_path = f"/uploads/crops/{crop_filename}"

            detection_response.append({
                "id": int(detection.id),
                "bbox": [
                    int(detection.bbox[0]),
                    int(detection.bbox[1]),
                    int(detection.bbox[2]),
                    int(detection.bbox[3])
                ],
                "confidence": float(detection.confidence),
                "ocr_confidence": float(cleaned.get("confidence", 0.0)),
                "class_name": str(detection.class_name),
                "class": f"{cleaned.get('text', '')} ({parsed.get('currency', 'Rs')}{parsed.get('price', '')})" if parsed.get('price') else cleaned.get('text', str(detection.class_name)),
                "text": cleaned.get("text", ""),
                "cropped_image_path": cropped_image_path,
                "price": parsed.get("price"),
                "currency": parsed.get("currency"),
                "status": business.get("status")
            })

            # draw detection
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            price = parsed.get("price")
            label = f"{parsed.get('currency', 'Rs')} {price:.2f}" if price is not None else "PRICE TAG"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # encode annotated image as base64 PNG
        _, buffer = cv2.imencode('.png', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "total_price_tags": len(detections),
            "detections": detection_response,
            "ocr_results": ocr_results,
            "cleaned_results": cleaned_results,
            "parsed_results": parsed_results,
            "business_results": business_results,
            "processed_crops": processed_crops,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
        }
