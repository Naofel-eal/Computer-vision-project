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
        self.change_model_names()
        self.warm_up_file = "resources/kad1.jpg"
        self.warm_up()
        
    def change_model_names(self):
        for index, name in enumerate(self.model.names):
            self.model.names[index] = f'person{index}'
        
    @abstractmethod
    def warm_up(self) -> None:
        pass
