import cv2

from models.interaction.PersonDetector import PersonDetector


class InteractionService:

    def __init__(self):
        self.detector = PersonDetector()

    def process(self, frame):

        results = self.detector.detect(frame)

        customers = []

        person_id = 1

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                customers.append({

                    "id": person_id,

                    "bbox": [x1, y1, x2, y2],

                    "confidence": round(confidence, 2)

                })

                person_id += 1

        return {

            "success": True,

            "total_customers": len(customers),

            "customers": customers

        }