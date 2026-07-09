import cv2

from models.detection.predict import ShelfDetector


class ShelfService:

    def __init__(self):

        self.detector = ShelfDetector()

    def process_frame(self, frame):

        results = self.detector.detect(frame)

        shelves = []

        shelf_id = 1

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                shelves.append({
                    "id": shelf_id,
                    "coordinates": [x1, y1, x2, y2]
                })

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Shelf {shelf_id}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2
                )

                shelf_id += 1

        return frame, shelves