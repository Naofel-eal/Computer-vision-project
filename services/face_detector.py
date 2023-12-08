from ultralytics import YOLO
from torch import cuda
from models.point import Point
from models.bounding_box import BoundingBox
from numpy import ndarray

from models.prediction import Prediction

class FaceDetector:
    def __init__(self, model_path: str = "weights/yolov8_detection_model.pt") -> None:
        self.model = YOLO(model_path)
        device = 'cuda' if cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("WARNING: Using CPU for face detection. This will be very slow.")
        self.model.to(device)
        self.warm_up()

    def warm_up(self) -> None:
        self.model.predict("resources/kad.jpg", verbose=False, save=False, show=False)
        print("Face detector warmed up.")

    def detect(self, img: ndarray) -> list[Prediction]:
        results = self.model.predict(img, conf=0.5,show=False, save=False, verbose=False)
        predictions = []
        for box_data in results[0].boxes.data.tolist():
            prediction = Prediction(BoundingBox(Point(box_data[0], box_data[1]), Point(box_data[2], box_data[3])), box_data[4])
            predictions.append(prediction)
        return predictions