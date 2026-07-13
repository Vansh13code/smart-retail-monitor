from ultralytics import YOLO
import logging
from pathlib import Path


def _find_pt_weights(preferred: list = None) -> str:
    """Search the repository for .pt weight files and return a sensible default path.

    Preference order:
    1. provided preferred list (ordered)
    2. runs/detect/*/weights/best.pt
    3. models/detection/*.pt
    4. models/detection/yolov8n.pt
    5. any .pt under project
    Returns None if nothing found.
    """
    root = Path(__file__).parents[2]
    preferred = preferred or []

    # check preferred list
    for p in preferred:
        pth = (root / p).resolve()
        if pth.exists():
            return str(pth)

    # runs/detect/*/weights/best.pt
    candidates = list(root.glob('runs/detect/**/weights/best.pt'))
    if candidates:
        return str(candidates[0].resolve())

    # models/detection/*.pt
    det_dir = root / 'models' / 'detection'
    candidates = list(det_dir.glob('*.pt')) if det_dir.exists() else []
    if candidates:
        for c in candidates:
            if 'yolov8n' in c.name.lower():
                return str(c.resolve())
        return str(candidates[0].resolve())

    # models/detection/yolov8n.pt
    alt = det_dir / 'yolov8n.pt'
    if alt.exists():
        return str(alt.resolve())

    # any .pt under project
    all_pts = list(root.rglob('*.pt'))
    if all_pts:
        return str(all_pts[0].resolve())

    return None


class ShelfDetector:

    def __init__(self, weights: str = None):
        self.weights = weights
        if weights:
            if Path(weights).exists():
                self.model = YOLO(weights)
                logging.info('Loaded shelf detector weights: %s', weights)
            else:
                logging.warning('Provided shelf weights not found: %s', weights)
                self.model = None
        else:
            found = _find_pt_weights(preferred=['models/detection/yolov8n.pt'])
            if found:
                try:
                    self.model = YOLO(found)
                    logging.info('Auto-loaded shelf detector weights: %s', found)
                except Exception as e:
                    logging.exception('Failed to load YOLO weights %s: %s', found, e)
                    self.model = None
            else:
                logging.warning('No YOLO weights found for ShelfDetector; detector disabled.')
                self.model = None

    def detect(self, frame, conf: float = 0.15, iou: float = 0.45, imgsz: int = 640):
        # If model couldn't be loaded, return empty results list
        if self.model is None:
            logging.warning('ShelfDetector.detect called but no model available; returning empty results')
            return []

        results = self.model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            save=False
        )

        logging.debug('ShelfDetector.detect: conf=%s iou=%s imgsz=%s -> boxes_per_result=%s', conf, iou, imgsz, [len(r.boxes) for r in results])

        return results