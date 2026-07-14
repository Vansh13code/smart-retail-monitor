"""
OCR Cache - Cache OCR results to avoid redundant processing
"""
import hashlib
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class OCRCache:
    """Simple in-memory cache for OCR results based on image hash"""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._access_order = []  # Track access order for LRU eviction
    
    def _compute_hash(self, image: np.ndarray) -> str:
        """Compute a hash of the image for cache key"""
        # Use a simpler hash for performance
        h, w = image.shape[:2]
        # Sample pixels for faster hashing
        sample_rate = max(1, (h * w) // 1000)
        sampled = image.flatten()[::sample_rate]
        return hashlib.md5(sampled.tobytes()).hexdigest()
    
    def get(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get cached OCR result for an image"""
        key = self._compute_hash(image)
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"OCR cache hit for key {key}")
            return self._cache[key]
        return None
    
    def set(self, image: np.ndarray, result: Dict[str, Any]):
        """Cache OCR result for an image"""
        key = self._compute_hash(image)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"OCR cache evicted key {oldest_key}")
        
        self._cache[key] = result
        self._access_order.append(key)
        logger.debug(f"OCR cache stored result for key {key}")
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("OCR cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": 0.0  # Would need to track hits/misses
        }


# Global OCR cache instance
ocr_cache = OCRCache(max_size=50)
