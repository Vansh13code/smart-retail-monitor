import cv2
import numpy as np
import base64
import logging

from models.interaction.PersonDetector import PersonDetector

logger = logging.getLogger(__name__)


class InteractionService:

    def __init__(self):
        self.detector = PersonDetector("yolo26n.pt")
        # Track history for sequential video frames
        # track_id -> list of [x, y] centers
        self.track_history = {}
        # track_id -> start_time
        self.track_start_times = {}
        # track_id -> last_seen_time
        self.track_last_seen = {}
        self.entries = set()
        self.exits = set()
        # Track unique customers seen
        self.unique_customers = set()

    def process(self, frame, timestamp: float = 0.0):
        # Reset trackers if we start a new video or process a single frame
        if timestamp == 0.0:
            self.track_history = {}
            self.track_start_times = {}
            self.track_last_seen = {}
            self.entries = set()
            self.exits = set()
            self.unique_customers = set()

        # Determine image dimensions
        h, w = frame.shape[:2]

        # Run tracking if possible
        try:
            results = self.detector.track(frame)
        except Exception as e:
            logger.warning("Tracking failed, falling back to detection: %s", e)
            try:
                results = self.detector.detect(frame)
            except Exception as e:
                logger.error("Detection also failed: %s", e)
                results = []

        customers = []

        # Parse results
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Check for tracking ID from Ultralytics tracker
                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    # Fallback: use detection index as temporary ID
                    track_id = idx + 1
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Update history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                    self.track_start_times[track_id] = timestamp
                    self.unique_customers.add(track_id)
                
                self.track_history[track_id].append([cx, cy])
                self.track_last_seen[track_id] = timestamp

                # Limit history size to 50 points for better path visualization
                if len(self.track_history[track_id]) > 50:
                    self.track_history[track_id].pop(0)

                # Calculate dwell time - only if we have actual timestamp data
                if timestamp > 0.0 and self.track_start_times[track_id] > 0:
                    dwell = round(self.track_last_seen[track_id] - self.track_start_times[track_id], 1)
                else:
                    # For single image, dwell time is 0 (no time data available)
                    dwell = 0.0

                # Determine path
                path = self.track_history[track_id]
                # For single image, use current position as path
                if len(path) == 1:
                    path = [[cx, cy]]

                customers.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 2),
                    "dwell_time": dwell,
                    "movement_path": path,
                    "first_seen": self.track_start_times[track_id],
                    "last_seen": self.track_last_seen[track_id]
                })

                # Determine entry / exit based on path vector
                if len(path) >= 2:
                    first_cx = path[0][0]
                    current_cx = path[-1][0]
                    
                    # Moving inwards from the edges: Entry
                    if first_cx < w * 0.3 and current_cx >= w * 0.3:
                        self.entries.add(track_id)
                    elif first_cx > w * 0.7 and current_cx <= w * 0.7:
                        self.entries.add(track_id)
                        
                    # Moving outwards to the edges: Exit
                    if first_cx >= w * 0.3 and first_cx <= w * 0.7:
                        if current_cx < w * 0.2 or current_cx > w * 0.8:
                            self.exits.add(track_id)
                else:
                    # Single frame - determine based on position
                    if cx < w * 0.3 or cx > w * 0.7:
                        self.entries.add(track_id)

        # Calculate entry / exit counts
        entry_count = len(self.entries)
        exit_count = len(self.exits)
        total_unique = len(self.unique_customers)

        # Draw annotated image
        annotated = frame.copy()
        for c in customers:
            x1, y1, x2, y2 = c["bbox"]
            # Blue bounding box for customers
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Label with customer ID and confidence
            label = f"Customer {c['id']} ({c['confidence']:.2f})"
            if c["dwell_time"] > 0:
                label += f" {c['dwell_time']}s"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (255, 0, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw movement path line
            path = c["movement_path"]
            if len(path) > 1:
                for i in range(len(path) - 1):
                    cv2.line(annotated, tuple(path[i]), tuple(path[i+1]), (0, 255, 255), 2)
                cv2.circle(annotated, tuple(path[-1]), 4, (0, 0, 255), -1)

        # Generate Heatmap
        heatmap_frame = frame.copy()
        overlay = np.zeros_like(frame, dtype=np.uint8)
        for c in customers:
            x1, y1, x2, y2 = c["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Red circles for current positions
            cv2.circle(overlay, (cx, cy), 60, (0, 0, 255), -1)
            # Green circles for path history
            for pt in c["movement_path"][:-1]:
                cv2.circle(overlay, tuple(pt), 30, (0, 255, 0), -1)

        if len(customers) > 0:
            blurred_overlay = cv2.GaussianBlur(overlay, (75, 75), 0)
            heatmap_frame = cv2.addWeighted(heatmap_frame, 0.7, blurred_overlay, 0.3, 0)

        # Encode images to base64
        _, buffer_ann = cv2.imencode('.png', annotated)
        annotated_b64 = base64.b64encode(buffer_ann).decode('utf-8')

        _, buffer_heat = cv2.imencode('.png', heatmap_frame)
        heatmap_b64 = base64.b64encode(buffer_heat).decode('utf-8')

        return {
            "success": True,
            "customer_count": len(customers),
            "total_customers": total_unique,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "customers": customers,
            "annotated_frame": annotated,  # Backwards compatibility for scripts
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "heatmap": f"data:image/png;base64,{heatmap_b64}",
            "tracking_enabled": True,
            "tracking_method": "ultralytics"
        }
