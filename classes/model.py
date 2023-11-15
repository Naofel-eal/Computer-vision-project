from ultralytics import YOLO
from torch import cuda

class Model:
    def __init__(self, model_path: str = "model/yolov8/model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)

    def detect(self, img):
        result = self.model.predict(img, show=True, save=False)
        bounding_boxes = result[0].boxes.xyxy.tolist()