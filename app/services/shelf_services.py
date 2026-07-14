import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

from models.detection.predict import ShelfDetector

logger = logging.getLogger(__name__)


class ShelfService:

    def __init__(self):
        try:
            self.detector = ShelfDetector()
            logger.info("ShelfService initialized successfully")
        except Exception as e:
            # If detector initialization fails, keep a disabled detector that returns empty results
            logger.exception('Failed to initialize ShelfDetector, disabling shelf detection: %s', e)
            class _Disabled:
                def detect(self, frame):
                    return []
            self.detector = _Disabled()

    def _iou(self, a: List[int], b: List[int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _nms(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.35) -> List[Dict[str, Any]]:
        if not detections:
            return []
        
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = []
        
        while sorted_dets:
            curr = sorted_dets.pop(0)
            keep.append(curr)
            remaining = []
            
            for d in sorted_dets:
                # Calculate IOU
                iou_val = self._iou(curr["bbox"], d["bbox"])
                
                # Check for nested containment (if box d is inside curr)
                cx1, cy1, cx2, cy2 = curr["bbox"]
                dx1, dy1, dx2, dy2 = d["bbox"]
                
                # Area of d
                area_d = (dx2 - dx1) * (dy2 - dy1)
                
                # Intersection area
                x1 = max(cx1, dx1)
                y1 = max(cy1, dy1)
                x2 = min(cx2, dx2)
                y2 = min(cy2, dy2)
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                
                containment = inter_area / area_d if area_d > 0 else 0.0
                
                # If they are both products and overlap significantly or one contains the other, skip d
                product_classes = ("shampoo", "milk", "snacks", "soft_drink", "soft drink", "bread", "beverages", "chocolate", "juice", "biscuits", "rice", "oil", "soap", "detergent", "toothpaste")
                is_curr_product = curr["class"].lower() in product_classes
                is_d_product = d["class"].lower() in product_classes
                
                if is_curr_product and is_d_product:
                    # High IOU or high containment means duplicate product box
                    if iou_val >= iou_threshold or containment > 0.8:
                        continue
                elif curr["class"].lower() == d["class"].lower():
                    # Class-wise NMS for shelves or other items
                    if iou_val >= iou_threshold:
                        continue
                        
                remaining.append(d)
                
            sorted_dets = remaining
            
        return keep

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

        # The supplied retail model produces useful detections around 0.10-0.16
        # on small web uploads.  The old 0.20 cutoff discarded every one of
        # them before ProductService ever received the objects.
        results = self.detector.detect(frame, conf=0.07, iou=0.40)
        logger.info("YOLO raw inference: threshold=0.07 model=%s", getattr(self.detector, "weights", "best.pt"))

        # If detector returned an empty list, normalize to empty list
        if results is None:
            results = []

        detections = []

        object_id = 1

        boxes = []

        # If results look like ultralytics results, iterate accordingly; else if it's already a list of dicts, use as-is
        if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'boxes'):
            for result in results:
                class_names = getattr(self.detector.model, 'names', None) or getattr(result, 'names', None)
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)

                    # Keep the lower global proposal threshold so small shelf
                    # labels are found, but reject noisy boxes class-wise.
                    minimum = 0.10 if str(class_name).strip().lower() == "shelf" else 0.12
                    if confidence < minimum:
                        continue
                    
                    # Extract tracking ID if available
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    detections.append({
                        "id": object_id,
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 2),
                        "track_id": track_id if track_id else object_id
                    })
                    boxes.append((x1, y1, x2, y2))
                    object_id += 1
        else:
            # results may already be a simple list of detection dicts
            try:
                for r in results:
                    if isinstance(r, dict) and 'bbox' in r:
                        detections.append(r)
                        boxes.append(tuple(r['bbox']))
                        object_id += 1
            except Exception:
                pass

        # Apply class-wise NMS to remove duplicates
        detections = self._nms(detections)

        logger.info(
            "YOLO handoff: detections=%d shelves=%d products=%d",
            len(detections),
            sum(d["class"].strip().lower() == "shelf" for d in detections),
            sum(d["class"].strip().lower() != "shelf" for d in detections),
        )

        # Reassign sequential IDs
        for i, det in enumerate(detections, start=1):
            det["id"] = i

        # Compute rows (if shelves present)
        shelf_boxes = [d["bbox"] for d in detections if d["class"].lower() == "shelf"]
        rows = []
        if shelf_boxes:
            row_clusters = self._cluster_rows(shelf_boxes)
            for r_idx, cluster in enumerate(row_clusters, start=1):
                row_boxes = [shelf_boxes[i] for i in cluster]
                rows.append({
                    "row_id": r_idx,
                    "shelves": row_boxes,
                    "shelf_count": len(row_boxes)
                })

        # Only shelves belong to this stage. Products are overlaid in blue by
        # ProductService's route so the colours stay semantically correct.
        annotated = frame.copy()
        for det in detections:
            if det["class"].strip().lower() != "shelf":
                continue
            x1, y1, x2, y2 = det["bbox"]
            
            # Green bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with class name, confidence, and tracking ID
            label = f"{det['class']} ({det['confidence']:.2f})"
            if det.get("track_id"):
                label += f" ID:{det['track_id']}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated, detections
    
    def annotate_with_inventory(self, frame, detections, shelf_inventory, ocr_data=None):
        """
        Enhanced annotation with inventory information, prices, and stock status
        """
        annotated = frame.copy()
        
        # Create price lookup from OCR data
        price_lookup = {}
        if ocr_data and "detections" in ocr_data:
            for detection in ocr_data["detections"]:
                bbox = detection.get("bbox")
                price = detection.get("price")
                if bbox and price is not None:
                    price_lookup[tuple(bbox)] = price
        
        # Annotate each detection with full information
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # Determine stock status based on shelf inventory
            stock_status = "Unknown"
            shelf_num = "N/A"
            prod_num = "N/A"
            
            for shelf in shelf_inventory:
                for idx, prod in enumerate(shelf.get("products", [])):
                    if prod.get("id") == det.get("id") or prod.get("track_id") == det.get("track_id"):
                        shelf_num = str(shelf.get("shelf_id", "N/A"))
                        prod_num = str(idx + 1)
                        
                        # Determine stock status
                        prod_count = shelf.get("product_count", 0)
                        if prod_count == 0:
                            stock_status = "Out of Stock"
                        elif prod_count <= 3:
                            stock_status = "Low Stock"
                        else:
                            stock_status = "In Stock"
                        break
            
            # Get price from OCR if available
            price = None
            for bbox, p in price_lookup.items():
                # Simple overlap check
                if (x1 <= bbox[0] <= x2 and y1 <= bbox[1] <= y2):
                    price = p
                    break
            
            # Color based on stock status
            if stock_status == "Out of Stock":
                box_color = (0, 0, 255)  # Red
            elif stock_status == "Low Stock":
                box_color = (0, 165, 255)  # Orange
            else:
                box_color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Build comprehensive label
            label_lines = [
                f"{det['class']} ({det['confidence']:.2f})",
                f"ID:{det.get('track_id', det['id'])}",
                f"Shelf:{shelf_num} Prod:{prod_num}",
                f"Status:{stock_status}"
            ]
            
            if price is not None:
                label_lines.append(f"Price:₹{price}")
            
            # Draw multi-line label
            y_offset = y1 - 5
            for i, line in enumerate(label_lines):
                (label_w, label_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (x1, y_offset - label_h - 2), (x1 + label_w + 4, y_offset + 2), box_color, -1)
                cv2.putText(annotated, line, (x1 + 2, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                y_offset -= label_h + 4
        
        return annotated
