from pathlib import Path
from ultralytics import YOLO
from app.core.model_manager import model_manager
import logging

from models.detection.predict import get_yolo_model

logger = logging.getLogger(__name__)


class PersonDetector:

    def __init__(self, weights: str = None):
        self.weights = weights
        self.model_warning = None
        try:
            self.model = get_yolo_model(weights)
            logging.info('InteractionDetector initialized using shared model loader.')
        except Exception as e:
            logging.exception('Failed to load YOLO weights %s: %s', weights, e)
            self.model = None
            raise e

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
