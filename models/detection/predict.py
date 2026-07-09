from ultralytics import YOLO

class ShelfDetector:

    def __init__(self):
        self.model = YOLO("runs/detect/shelf_detector2/weights/best.pt")

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=0.5,
            verbose=False
        )

        return results