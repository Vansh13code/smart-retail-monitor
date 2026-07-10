from dataclasses import dataclass
import numpy as np


@dataclass
class Crop:

    id: int

    image: np.ndarray

    bbox: tuple[int, int, int, int]

    confidence: float