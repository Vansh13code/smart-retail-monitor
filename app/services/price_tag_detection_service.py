from typing import List

import cv2
import easyocr
import re

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.schemas.detection import Detection
from app.schemas.crop import Crop


# ==========================================================
# Price Tag Detector
# ==========================================================

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


# ==========================================================
# Cropper
# ==========================================================

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

        # Convert to grayscale
        gray = cv2.cvtColor(
            crop.image,
            cv2.COLOR_BGR2GRAY
        )

        # Improve local contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # Remove small noise
        denoised = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0
        )

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return enhanced
    

# ==========================================================
# OCR Service
# ==========================================================

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
            text_threshold=0.4,
            low_text=0.2
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
    

# ==========================================================
# Text Cleaner
# ==========================================================

class TextCleaner:
    """
    Cleans and normalizes OCR text before parsing.
    """

    def clean(self, text: str) -> str:

        if not text:
            return ""

        # ----------------------------------------
        # Convert to uppercase
        # ----------------------------------------

        text = text.upper()

        # ----------------------------------------
        # Replace common OCR mistakes
        # ----------------------------------------

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

        # ----------------------------------------
        # Remove unwanted symbols
        # Keep letters, digits, ₹, dots and spaces
        # ----------------------------------------

        text = re.sub(
            r"[^A-Z0-9₹.\s]",
            " ",
            text
        )

        # ----------------------------------------
        # Remove multiple spaces
        # ----------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()



# ==========================================================
# Text Parser
# ==========================================================

class TextParser:
    """
    Parses OCR text and extracts useful information.
    """

    def parse(self, ocr_result):
        text = ocr_result["text"]

        # ----------------------------------------
        # Find price
        # ----------------------------------------

        price_pattern = r"\d+(?:\.\d{1,2})?"

        prices = re.findall(
            price_pattern,
            text
        )

        price = None
        if prices:
            price = float(prices[0])

        # ----------------------------------------
        # Clean product text
        # ----------------------------------------

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



# ==========================================================
# Price Tag Service
# ==========================================================

class PriceTagService:

    def __init__(self):

        self.detector = PriceTagDetector()

        self.cropper = Cropper()

        self.preprocessor = ImagePreprocessor()

        self.ocr = OCRService()

        self.cleaner = TextCleaner()

        self.parser = TextParser()


    def process(self, frame):

        # --------------------------------------------
        # Step 1 : Detect Price Tags
        # --------------------------------------------

        detections = self.detector.detect(frame)

        # --------------------------------------------
        # Step 2 : Crop Tags
        # --------------------------------------------

        crops = self.cropper.crop(
            frame,
            detections
        )

        # --------------------------------------------
        # Step 3 : Preprocess
        # --------------------------------------------

        processed_crops = []
        for crop in crops:
            processed = self.preprocessor.preprocess(crop)
            processed_crops.append(processed)



        # --------------------------------------------
        # Step 4 : OCR
        # --------------------------------------------

        ocr_results = []
        for processed in processed_crops:
            result = self.ocr.read(processed)
            ocr_results.append(result)


        cleaned_results = []
        for result in ocr_results:
            cleaned_text = self.cleaner.clean(
                result["text"]
            )
            cleaned_results.append({
                "text": cleaned_text,
                "confidence": result["confidence"]
            })



        parsed_results = []
        for result in ocr_results:
            parsed = self.parser.parse(result)
            parsed_results.append(parsed)



        # --------------------------------------------
        # Return
        # --------------------------------------------

        return {
            "detections": detections,
            
            "crops": crops,
            
            "processed_crops": processed_crops,
            
            "ocr_results": ocr_results,
            
            "cleaned_results": cleaned_results,

            "parsed_results": parsed_results,

            "total_price_tags": len(detections)
            
        }