from typing import List

import cv2
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

        return binary
    
# ==========================================================
# Price Tag Service
# ==========================================================

class PriceTagService:
    """
    Complete Price Tag Detection Pipeline

    Pipeline:
        Image
          ↓
        YOLO Detection
          ↓
        Crop
          ↓
        Preprocess
    """

    def __init__(self):

        self.detector = PriceTagDetector()
        self.cropper = Cropper()
        self.preprocessor = ImagePreprocessor()

    def process(self, frame):

        # Step 1 : Detect price tags
        detections = self.detector.detect(frame)

        # Step 2 : Crop detected price tags
        crops = self.cropper.crop(
            frame,
            detections
        )

        # Step 3 : Preprocess each crop
        processed_crops = []

        for crop in crops:

            processed = self.preprocessor.preprocess(crop)

            processed_crops.append(processed)

        return {

            "detections": detections,

            "crops": crops,

            "processed_crops": processed_crops,

            "total_price_tags": len(detections)

        }