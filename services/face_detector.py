from ultralytics import YOLO
from torch import cuda
from models.point import Point
from models.bounding_box import BoundingBox
from numpy import ndarray

from models.prediction import Prediction
from utils.performance_counter import PerformanceCounter

class FaceDetector:
    def __init__(self, model_path: str = "weights/model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)

    def detect(self, img: ndarray) -> list[Prediction]:
        results = self.model.predict(img, show=False, save=False, verbose=False)
        predictions = []
        for box_data in results[0].boxes.data.tolist():
            prediction = Prediction(BoundingBox(Point(box_data[0], box_data[1]), Point(box_data[2], box_data[3])), box_data[4])
            predictions.append(prediction)
        return predictions