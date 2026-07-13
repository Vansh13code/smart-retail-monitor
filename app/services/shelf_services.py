import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

from models.detection.predict import ShelfDetector


class ShelfService:

    def __init__(self):
        try:
            self.detector = ShelfDetector()
        except Exception as e:
            # If detector initialization fails, keep a disabled detector that returns empty results
            import logging
            logging.exception('Failed to initialize ShelfDetector, disabling shelf detection: %s', e)
            class _Disabled:
                def detect(self, frame):
                    return []
            self.detector = _Disabled()

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

        # Run YOLO detection (detector may return list or ultralytics results)
        results = self.detector.detect(frame)

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
                    detections.append({
                        "id": object_id,
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 2)
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

        # Annotate image
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det['class']} ({det['confidence']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated, detections