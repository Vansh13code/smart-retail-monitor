from ultralytics import YOLO
import logging
import os
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

    def score_model(pth) -> int:
        try:
            model = YOLO(str(pth))
            names = model.names if hasattr(model, 'names') else {}
            model_labels = list(names.values()) if isinstance(names, dict) else list(names)
            normalized = {str(name).strip().lower().replace('-', ' ').replace('_', ' ') for name in model_labels}
            expected = {'shelf', 'shampoo', 'milk', 'snacks', 'soft drink'}
            return len(expected.intersection(normalized))
        except Exception:
            return -1

    # runs/detect/*/weights/best.pt
    candidates = list(root.glob('runs/detect/**/weights/best.pt'))
    if candidates:
        candidates = sorted(candidates, key=lambda c: (score_model(c), str(c)), reverse=True)
        return str(candidates[0].resolve())

    # models/detection/*.pt
    det_dir = root / 'models' / 'detection'
    candidates = list(det_dir.glob('*.pt')) if det_dir.exists() else []
    candidates = [c for c in candidates if 'yolov8n' not in c.name.lower()]
    if candidates:
        candidates = sorted(candidates, key=lambda c: (score_model(c), str(c)), reverse=True)
        return str(candidates[0].resolve())

    # models/detection/yolov8n.pt
    alt = det_dir / 'yolov8n.pt'
    if alt.exists():
        return str(alt.resolve())

    # any .pt under project
    all_pts = list(root.rglob('*.pt'))
    all_pts = [p for p in all_pts if '.venv' not in p.parts and '.git' not in p.parts]
    if all_pts:
        all_pts = sorted(all_pts, key=lambda c: (score_model(c), str(c)), reverse=True)
        return str(all_pts[0].resolve())

    return None


class ShelfDetector:

    def __init__(self, weights: str = None):
        self.weights = weights
        self.model_warning = None
        if weights:
            if Path(weights).exists():
                self.model = YOLO(weights)
                logging.info('Loaded shelf detector weights: %s', weights)
            else:
                logging.warning('Provided shelf weights not found: %s', weights)
                self.model = None
        else:
            env_weights = os.getenv('SMART_RETAIL_WEIGHTS')
            preferred = [env_weights] if env_weights else []
            found = _find_pt_weights(preferred=preferred)
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

        if self.model is not None:
            names = self.model.names if hasattr(self.model, 'names') else {}
            model_labels = list(names.values()) if isinstance(names, dict) else list(names)
            normalized = {str(name).strip().lower().replace('-', ' ').replace('_', ' ') for name in model_labels}
            expected = {'shelf', 'shampoo', 'milk', 'snacks', 'soft drink'}
            if not expected.intersection(normalized):
                self.model_warning = (
                    'Retail classes were not found in model labels. '
                    'Current weights look generic; product/classification counts may be inaccurate.'
                )
                logging.warning(self.model_warning)

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