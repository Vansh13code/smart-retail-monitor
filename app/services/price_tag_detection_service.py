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

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.schemas.detection import Detection
from app.schemas.crop import Crop


class PriceTagDetector:
    """
    Detects price tags using a pretrained YOLO model.
    """

    def __init__(self):

        logging.info("Loading Price Tag Detection Model...")

        model_path = hf_hub_download(
            repo_id="openfoodfacts/price-tag-detection",
            filename="weights/best.pt"
        )

        # Let Ultralytics choose device automatically, but record availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

        logging.info("Price tag detection model loaded on %s", self.device)

    def detect(self, image, conf: float = 0.25, iou: float = 0.45) -> List[Detection]:

        # Run inference (Ultralytics applies NMS internally)
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    id=idx,
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=self.model.names[int(box.cls[0])]
                )
            )

        # Merge highly overlapping boxes to reduce duplicates
        merged = self._merge_overlapping(detections)
        return merged

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

            roi = image[y1:y2, x1:x2]

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
    Preprocess cropped price tag images to improve OCR accuracy.
    """

    def preprocess(self, crop: Crop) -> Any:
        img = crop.image.copy()

        # Upscale small crops to improve OCR on small text
        h, w = img.shape[:2]
        scale = 2 if max(h, w) < 200 else 1
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to gray and apply CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Gamma correction to handle lighting
        gamma = 1.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(sharpened, table)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 9)

        # Morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Deskew using moments / minAreaRect
        coords = np.column_stack(np.where(opened < 255))
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.1:
                (h, w) = opened.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                opened = cv2.warpAffine(opened, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return opened
    


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

    def read(self, image, min_confidence: float = 0.45):
        # image is expected to be a binary or gray image
        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.3,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            width_ths=0.7,
            ycenter_ths=0.5,
            height_ths=0.5,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789₹RsMRP.-/: "
        )

        # results -> list of (bbox, text, confidence)
        filtered = []
        for bbox, text, conf in results:
            if conf is None:
                continue
            if conf >= min_confidence and text.strip():
                filtered.append({"bbox": bbox, "text": text.strip(), "confidence": float(conf)})

        # Merge fragmented adjacent boxes (simple heuristic: close vertical overlap)
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
            abox = a["bbox"]
            used[i] = True
            for j, b in enumerate(filtered[i + 1 :], start=i + 1):
                if used[j]:
                    continue
                bx = np.array(b["bbox"]).reshape(-1, 2)
                bymin = bx[:, 1].min()
                bymax = bx[:, 1].max()
                # if vertical overlap significant or close
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

        return {"text": final_text, "confidence": float(avg_confidence), "blocks": merged}
    


class TextCleaner:
    """
    Cleans and normalizes OCR text before parsing.
    """

    def clean(self, text: str) -> str:

        if not text:
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

        text = re.sub(
            r"[^A-Z0-9₹.\s]",
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

        # Look for currency symbols + numbers
        price_pattern = r"(?:₹|Rs\.?|INR)?\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"

        matches = re.findall(price_pattern, text, flags=re.IGNORECASE)

        price = None
        currency = None
        if matches:
            # pick the largest plausible number (handles fragmented reads like 19 vs 199)
            numeric = [m.replace(",", "").replace(" ", "") for m in matches]
            numeric_f = [float(n) for n in numeric if re.match(r"^[0-9.]+$", n)]
            if numeric_f:
                price = float(max(numeric_f))
                # find currency symbol near the price
                cur_match = re.search(r"(₹|Rs\.?|INR)", text, flags=re.IGNORECASE)
                currency = cur_match.group(1) if cur_match else ""

        cleaned_text = re.sub(price_pattern, "", text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.replace("₹", "")
        cleaned_text = cleaned_text.strip()

        return {
            "product_name": cleaned_text,
            "price": price,
            "currency": currency,
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
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return {
                "total_price_tags": 0,
                "detections": [],
                "ocr_results": [],
                "cleaned_results": [],
                "parsed_results": [],
                "business_results": [],
                "annotated_image": None,
            }

        crops = self.cropper.crop(frame, detections)

        ocr_results = []
        cleaned_results = []
        parsed_results = []
        business_results = []
        detection_response = []

        # Annotate on a copy
        annotated = frame.copy()

        for detection, crop in zip(detections, crops):

            processed = self.preprocessor.preprocess(crop)

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

            detection_response.append({
                "id": int(detection.id),
                "bbox": [
                    int(detection.bbox[0]),
                    int(detection.bbox[1]),
                    int(detection.bbox[2]),
                    int(detection.bbox[3])
                ],
                "confidence": float(detection.confidence),
                "class_name": str(detection.class_name)
            })

            # draw detection
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{parsed.get('price', '')} {parsed.get('currency', '')} ({cleaned.get('confidence', 0):.2f})"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
        }