from numpy import ndarray, zeros
from ultralytics import YOLO
from torch import cuda
import logging

from models.point import Point
from models.bounding_box import BoundingBox
from models.prediction import Prediction
from services.faces.model import Model

class FaceDetector(Model):
    def __init__(self, model_path: str = "weights/yolov8_detection_model.pt") -> None:
        Model.__init__(self, model_path=model_path)
        self.model = YOLO(self.model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        logging.info(f"Yolo FaceDetector model running on: {device}")
        self.model.to(device)
        self.warm_up()

    def warm_up(self) -> None:
        empty_image = zeros((640, 640, 3))
        self.detect(empty_image)
        logging.info("Yolo FaceDetector warmed up.")

    def detect(self, img: ndarray) -> list[Prediction]:
        results = self.model.predict(img, conf=0.5,show=False, save=False, verbose=False)
        predictions = []
        for box_data in results[0].boxes.data.tolist():
            prediction = Prediction(BoundingBox(Point(box_data[0], box_data[1]), Point(box_data[2], box_data[3])), box_data[4])
            predictions.append(prediction)
        return predictions