from typing import List

import cv2
import easyocr
import re

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.schemas.detection import Detection
from app.schemas.crop import Crop


class PriceTagDetector:
    """
    Detects price tags using a pretrained YOLO model.
    """

    def __init__(self):

        print("Loading Price Tag Detection Model...")

        model_path = hf_hub_download(
            repo_id="openfoodfacts/price-tag-detection",
            filename="weights/best.pt"
        )

        self.model = YOLO(model_path)

        print("Model loaded successfully!")

    def detect(self, image, conf: float = 0.25) -> List[Detection]:

        results = self.model.predict(
            source=image,
            conf=conf,
            verbose=False
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
        return detections



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

    def preprocess(self, crop):

        crop.image = cv2.resize(
            crop.image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(
            crop.image,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=4.0,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)

        denoised = cv2.GaussianBlur(
            enhanced,
            (5, 5),
            0
        )

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5
        )

        return binary
    


class OCRService:
    """
    Extracts text from preprocessed price tag images
    using EasyOCR.
    """

    def __init__(self):

        print("Loading EasyOCR Model...")

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        print("EasyOCR loaded successfully!")

    def read(self, image):

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

            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789₹.-/: "
        )

        extracted_text = []
        confidences = []

        for _, text, confidence in results:
            extracted_text.append(text)
            confidences.append(confidence)

        final_text = " ".join(extracted_text)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        return {
            "text": final_text,
            "confidence": average_confidence
        }
    


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

    def parse(self, ocr_result):
        text = ocr_result["text"]

        price_pattern = r"\d+(?:\.\d{1,2})?"

        prices = re.findall(
            price_pattern,
            text
        )

        price = None
        if prices:
            price = float(prices[0])

        cleaned_text = re.sub(
            price_pattern,
            "",
            text
        )

        cleaned_text = cleaned_text.replace(
            "₹",
            ""
        )

        cleaned_text = cleaned_text.strip()

        return {
            "product_name": cleaned_text,
            "price": price,
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
                "business_results": []
            }

        crops = self.cropper.crop(frame, detections)

        ocr_results = []
        cleaned_results = []
        parsed_results = []
        business_results = []
        detection_response = []

        for detection, crop in zip(detections, crops):

            processed = self.preprocessor.preprocess(crop)

            ocr = self.ocr.read(processed)

            cleaned = {
                "text": self.cleaner.clean(ocr["text"]),
                "confidence": float(ocr["confidence"])
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

        return {

            "total_price_tags": len(detections),

            "detections": detection_response,

            "ocr_results": ocr_results,

            "cleaned_results": cleaned_results,

            "parsed_results": parsed_results,

            "business_results": business_results

        }