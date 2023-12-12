from ultralytics import YOLO
from torch import cuda

class FaceModel:
    def __init__(self, model_path: str, model_type: str) -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("WARNING: Using CPU for face detection. This will be very slow.")
        self.model.to(device)
        self.warm_up(model_type=model_type)
        
    def warm_up(self, model_type: str) -> None:
        self.model.predict("resources/kad1.jpg", verbose=False, save=False, show=False)

        if(model_type == 'detector'):
            print("Face detector warmed up.")

        elif(model_type == 'comparator'):
            print("Face comparator warmed up.")
