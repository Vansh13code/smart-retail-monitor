from dataclasses import dataclass

@dataclass
class Detection:
    id: int
    bbox: tuple
    confidence: float
    class_id: int
    class_name: str