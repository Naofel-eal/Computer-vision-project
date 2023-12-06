from models.bounding_box import BoundingBox


class Prediction:
    def __init__(self, bounding_box: BoundingBox, confidence: float) -> None:
        self.bounding_box: BoundingBox = bounding_box
        self.confidence: float = confidence

    def __str__(self) -> str:
        return f"Prediction: {self.bounding_box}, {self.confidence*100:.2f}%"
    
    def __eq__(self, other: 'Prediction') -> bool: 
        if not isinstance(other, Prediction):
            return NotImplemented

        return self.bounding_box == other.bounding_box and self.confidence == other.confidence

