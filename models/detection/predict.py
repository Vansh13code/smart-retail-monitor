from ultralytics import YOLO

class ShelfDetector:

    def __init__(self):
        self.model = YOLO("runs/detect/shelf_detector3/weights/best.pt")

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=0.15,
            imgsz=640,
            verbose=True,
            save=False
        )

        print("=" * 60)

        for r in results:
            print("Model Classes:", self.model.names)
            print("Detected Boxes:", len(r.boxes))

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Class: {self.model.names[cls]}  Confidence: {conf:.3f}")

        print("=" * 60)

        return results