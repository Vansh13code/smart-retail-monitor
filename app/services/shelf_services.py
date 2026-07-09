import cv2

from models.detection.predict import ShelfDetector


class ShelfService:

    def __init__(self):
        self.detector = ShelfDetector()

    def process_frame(self, frame):

        # Run YOLO detection
        results = self.detector.detect(frame)

        detections = []

        object_id = 1

        for result in results:

            class_names = result.names

            for box in result.boxes:

                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence
                confidence = float(box.conf[0])

                # Class ID
                class_id = int(box.cls[0])

                # Class Name
                class_name = class_names[class_id]

                # Save detection
                detections.append({

                    "id": object_id,

                    "class": class_name,

                    "bbox": [x1, y1, x2, y2],

                    "confidence": round(confidence, 2)

                })

                # Draw Rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw Label
                cv2.putText(
                    frame,
                    f"{class_name} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                object_id += 1

        return frame, detections