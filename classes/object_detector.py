from ultralytics import YOLO
from torch import cuda

class ObjectDetector:
    def __init__(self, model_path: str = "model/yolov8/model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)

    def detect(self, img):
        result = self.model.predict(img, show=True, save=False, verbose=False)
        bounding_boxes = result[0].boxes.xyxy.tolist()

    def crop(self, img, bounding_boxes: list):
        #todo