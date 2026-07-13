from pathlib import Path
from ultralytics import YOLO


class PersonDetector:

    def __init__(self):

        model_path = Path(__file__).parent / "yolov8n.pt"

        self.model = YOLO(str(model_path))

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=0.4,
            verbose=False
        )

        return results

    def track(self, frame):

        results = self.model.track(
            source=frame,
            classes=[0],
            persist=True,
            conf=0.4,
            verbose=False
        )

        return results