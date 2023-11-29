from models.prediction import Prediction

class Face:
    
    def __init__(self, frame_index: int, prediction: Prediction, distance: float = 1.0) -> None:
        self.frame_index: int = frame_index
        self.prediction: Prediction = prediction
        self.distance: float = distance