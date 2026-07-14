from ultralytics import YOLO
import logging
from pathlib import Path
from app.core.model_manager import model_manager

logger = logging.getLogger(__name__)


def get_yolo_model(weights: str = None) -> YOLO:
    """Gets or loads a cached YOLO model using the centralized ModelManager."""
    if weights is None:
        # Use best.pt by default through ModelManager
        return model_manager.get_best_model()
    
    # If specific weights requested, check if it's best.pt
    root = Path(__file__).parents[2]
    pth = Path(weights)
    
    if pth.name == 'best.pt' or (root / weights).name == 'best.pt':
        return model_manager.get_best_model()
    
    # For other weights, use legacy loading
    global _model_cache
    if weights not in _model_cache:
        logger.info('Loading YOLO model from: %s', weights)
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


_model_cache = {}


class ShelfDetector:

    def __init__(self, weights: str = None):
        self.weights = weights
        self.model_warning = None
        try:
            self.model = get_yolo_model(weights)
            logging.info('ShelfDetector initialized using shared model loader.')
        except Exception as e:
            logging.exception('Failed to load YOLO weights %s: %s', weights, e)
            self.model = None
            raise e

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
