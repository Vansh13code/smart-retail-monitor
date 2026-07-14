from pathlib import Path
from ultralytics import YOLO
from app.core.model_manager import model_manager
import logging

logger = logging.getLogger(__name__)


class PersonDetector:

    def __init__(self):
        try:
            # Use centralized ModelManager to load best.pt
            self.model = model_manager.get_person_model()
            logger.info("PersonDetector initialized using centralized ModelManager")
        except Exception as e:
            logger.exception("Failed to load person model: %s", e)
            # Fallback to yolov8n.pt if best.pt doesn't have person class
            try:
                from models.detection.predict import get_yolo_model
                self.model = get_yolo_model("yolov8n.pt")
                logger.info("PersonDetector fell back to yolov8n.pt")
            except Exception as fallback_e:
                logger.exception("Fallback model loading also failed: %s", fallback_e)
                raise

    def _get_person_class_id(self):
        """Find the class ID for 'person' in the model"""
        for class_id, class_name in self.model.names.items():
            if str(class_name).lower() == "person":
                return class_id
        return None

    def detect(self, frame):
        person_class_id = self._get_person_class_id()
        
        results = self.model.predict(
            source=frame,
            classes=[person_class_id] if person_class_id is not None else None,
            conf=0.4,
            verbose=False
        )

        return results

    def track(self, frame):
        person_class_id = self._get_person_class_id()
        
        # Ultralytics track requires classes to exist
        classes_filter = [person_class_id] if person_class_id is not None else None
        results = self.model.track(
            source=frame,
            classes=classes_filter,
            persist=True,
            conf=0.4,
            verbose=False
        )

        return results
