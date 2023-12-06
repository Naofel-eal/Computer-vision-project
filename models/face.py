from models.prediction import Prediction

class Face:

    def __init__(self, frame_index: int, prediction: Prediction) -> None:
        self.frame_index: int = frame_index
        self.prediction: Prediction = prediction

    def __str__(self) -> str:
        return f"Face in frame {self.frame_index} - {self.prediction}"

    def __eq__(self, other: 'Face') -> bool: 
        if not isinstance(other, Face):
            return NotImplemented

        return self.frame_index == other.frame_index and self.prediction == other.prediction