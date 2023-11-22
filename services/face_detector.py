from ultralytics import YOLO
from torch import cuda
from models.point import Point
from models.bounding_box import BoundingBox
from numpy import ndarray

class FaceDetector:
    def __init__(self, model_path: str = "weights/model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)

    def detect(self, img: ndarray) -> list[BoundingBox]:
        results = self.model.predict(img, show=False, save=False, verbose=False)
        bounding_boxes = results[0].boxes.xyxy.tolist()
        bounding_boxes = [BoundingBox(Point(box[0], box[1]), Point(box[2], box[3])) for box in bounding_boxes]
        return bounding_boxes

    def crop(self, img: ndarray, bounding_boxes: list[BoundingBox]):
        pass