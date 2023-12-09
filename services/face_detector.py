from models.point import Point
from models.bounding_box import BoundingBox
from numpy import ndarray

from models.prediction import Prediction
from services.face_model import FaceModel

class FaceDetector(FaceModel):
    def __init__(self, model_path: str = "weights/yolov8_detection_model.pt") -> None:
        super().__init__(model_path, "detector")

    def detect(self, img: ndarray) -> list[Prediction]:
        results = self.model.predict(img, conf=0.5,show=False, save=False, verbose=False)
        predictions = []
        for box_data in results[0].boxes.data.tolist():
            prediction = Prediction(BoundingBox(Point(box_data[0], box_data[1]), Point(box_data[2], box_data[3])), box_data[4])
            predictions.append(prediction)
        return predictions