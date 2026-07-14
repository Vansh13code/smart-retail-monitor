"""
Central Model Manager - Load models once at startup with GPU support
"""
import torch
import logging
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager to load and cache YOLO models"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def get_best_model(self) -> YOLO:
        """Load or return cached best.pt model"""
        if "best" not in self._models:
            root = Path(__file__).parents[2]
            best_path = root / "best.pt"
            
            if not best_path.exists():
                raise FileNotFoundError(
                    f"Required model file 'best.pt' not found in {root}. "
                    "Please ensure best.pt exists in the project root."
                )
            
            logger.info(f"Loading best.pt from {best_path} on {self.device}")
            self._models["best"] = YOLO(str(best_path))
            self._models["best"].to(self.device)
            logger.info("best.pt loaded successfully")
        
        return self._models["best"]
    
    def get_person_model(self) -> YOLO:
        """Load or return cached person detection model (also uses best.pt)"""
        # Use best.pt for person detection too - it should have person class
        return self.get_best_model()
    
    def get_price_tag_model(self) -> YOLO:
        """Load or return cached price tag model (uses best.pt)"""
        return self.get_best_model()
    
    def preload_all_models(self):
        """Preload all models at startup"""
        logger.info("Preloading all models...")
        try:
            self.get_best_model()
            logger.info("All models preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
            raise
    
    def clear_cache(self):
        """Clear model cache (useful for testing)"""
        self._models.clear()
        logger.info("Model cache cleared")


# Global instance
model_manager = ModelManager()
