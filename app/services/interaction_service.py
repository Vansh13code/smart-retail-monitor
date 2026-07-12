import cv2

from models.interaction.PersonDetector import PersonDetector


class InteractionService:

    def __init__(self):

        self.detector = PersonDetector()

    def process(self, frame):

        results = self.detector.track(frame)

        customers = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # -----------------------------
                # Bounding Box
                # -----------------------------
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # -----------------------------
                # Confidence
                # -----------------------------
                confidence = float(box.conf[0])

                # -----------------------------
                # Tracking ID
                # -----------------------------
                if box.id is not None:
                    customer_id = int(box.id[0])
                else:
                    customer_id = -1

                # -----------------------------
                # Save Customer Information
                # -----------------------------
                customers.append({

                    "id": customer_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 2)

                })

                # -----------------------------
                # Draw Bounding Box
                # -----------------------------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                # -----------------------------
                # Draw Customer ID
                # -----------------------------
                cv2.putText(
                    frame,
                    f"Customer {customer_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        return {

            "annotated_frame": frame,
            "customers": customers

        }