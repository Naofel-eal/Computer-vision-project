from abc import abstractmethod
from ultralytics import YOLO
from torch import cuda

class YoloModel:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)        
        device = 'cuda' if cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("WARNING: Using CPU for face detection. Inferences will be very slow.")
        self.model.to(device)
        
    @abstractmethod
    def warm_up(self) -> None:
        pass
