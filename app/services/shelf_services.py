import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

from models.detection.predict import ShelfDetector

logger = logging.getLogger(__name__)


class ShelfService:

    def __init__(self):
        try:
            self.detector = ShelfDetector("shelf_best.pt")
        except Exception:
            logger.exception("Failed to initialize ShelfDetector")
            raise

    def _cluster_rows(self, boxes: List[Tuple[int, int, int, int]], threshold: int = 40):
        # Cluster shelf boxes by their vertical center to identify rows
        centers = [(b[1] + b[3]) // 2 for b in boxes]
        sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
        rows = []
        current = []
        for idx in sorted_idx:
            if not current:
                current = [idx]
                continue
            prev = current[-1]
            if abs(centers[idx] - centers[prev]) <= threshold:
                current.append(idx)
            else:
                rows.append(current)
                current = [idx]
        if current:
            rows.append(current)
        return rows

    def process_frame(self, frame) -> Tuple[Any, List[Dict[str, Any]]]:

        results = self.detector.detect(
            frame,
            conf=0.25,
            iou=0.40
        )

        if results is None:
            results = []

        detections = []
        object_id = 1

        if (
            isinstance(results, list)
            and len(results) > 0
            and hasattr(results[0], "boxes")
        ):

            for result in results:

                class_names = (
                    getattr(self.detector.model, "names", None)
                    or getattr(result, "names", None)
                )

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    confidence = float(box.conf[0])

                    class_id = int(box.cls[0])

                    class_name = (
                        class_names[class_id]
                        if class_names
                        and class_id in class_names
                        else str(class_id)
                    )

                    # Both dataset classes represent shelves
                    if str(class_name).strip() in {"1", "2"}:
                        class_name = "Shelf"

                    detections.append({
                        "id": object_id,
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 2),
                        "track_id": object_id
                    })

                    object_id += 1

        annotated = frame.copy()

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            # Green shelf bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"Shelf "
                f"({det['confidence']:.2f}) "
                f"ID:{det['id']}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        logger.info(
            "Shelf detection completed: shelves=%d",
            len(detections)
        )

        return annotated, detections
    
    # def annotate_with_inventory(self, frame, detections, shelf_inventory, ocr_data=None):
    #     """
    #     Enhanced annotation with inventory information, prices, and stock status
    #     """
    #     annotated = frame.copy()
        
    #     # Create price lookup from OCR data
    #     price_lookup = {}
    #     if ocr_data and "detections" in ocr_data:
    #         for detection in ocr_data["detections"]:
    #             bbox = detection.get("bbox")
    #             price = detection.get("price")
    #             if bbox and price is not None:
    #                 price_lookup[tuple(bbox)] = price
        
    #     # Annotate each detection with full information
    #     for det in detections:
    #         x1, y1, x2, y2 = det["bbox"]
            
    #         # Determine stock status based on shelf inventory
    #         stock_status = "Unknown"
    #         shelf_num = "N/A"
    #         prod_num = "N/A"
            
    #         for shelf in shelf_inventory:
    #             for idx, prod in enumerate(shelf.get("products", [])):
    #                 if prod.get("id") == det.get("id") or prod.get("track_id") == det.get("track_id"):
    #                     shelf_num = str(shelf.get("shelf_id", "N/A"))
    #                     prod_num = str(idx + 1)
                        
    #                     # Determine stock status
    #                     prod_count = shelf.get("product_count", 0)
    #                     if prod_count == 0:
    #                         stock_status = "Out of Stock"
    #                     elif prod_count <= 3:
    #                         stock_status = "Low Stock"
    #                     else:
    #                         stock_status = "In Stock"
    #                     break
            
    #         # Get price from OCR if available
    #         price = None
    #         for bbox, p in price_lookup.items():
    #             # Simple overlap check
    #             if (x1 <= bbox[0] <= x2 and y1 <= bbox[1] <= y2):
    #                 price = p
    #                 break
            
    #         # Color based on stock status
    #         if stock_status == "Out of Stock":
    #             box_color = (0, 0, 255)  # Red
    #         elif stock_status == "Low Stock":
    #             box_color = (0, 165, 255)  # Orange
    #         else:
    #             box_color = (0, 255, 0)  # Green
            
    #         # Draw bounding box
    #         cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
    #         # Build comprehensive label
    #         label_lines = [
    #             f"{det['class']} ({det['confidence']:.2f})",
    #             f"ID:{det.get('track_id', det['id'])}",
    #             f"Shelf:{shelf_num} Prod:{prod_num}",
    #             f"Status:{stock_status}"
    #         ]
            
    #         if price is not None:
    #             label_lines.append(f"Price:₹{price}")
            
    #         # Draw multi-line label
    #         y_offset = y1 - 5
    #         for i, line in enumerate(label_lines):
    #             (label_w, label_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    #             cv2.rectangle(annotated, (x1, y_offset - label_h - 2), (x1 + label_w + 4, y_offset + 2), box_color, -1)
    #             cv2.putText(annotated, line, (x1 + 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    #             y_offset -= label_h + 4
        
    #     return annotated
