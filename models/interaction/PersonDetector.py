from ultralytics import YOLO


class PersonDetector:

    def __init__(self):

        # Load YOLO model
        self.model = YOLO(r"C:\Users\DELL\OneDrive\Desktop\New folder (3)\smart-retail-monitor\models\interaction\yolov8n.pt")

    def track(self, frame):

        results = self.model.track(

            source=frame,

            classes=[0],          # Detect only 'person' class

            persist=True,         # Keep tracking IDs between frames

            conf=0.4,             # Confidence threshold

            verbose=False

        )

        return results