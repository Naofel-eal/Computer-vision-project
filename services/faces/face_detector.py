from models.point import Point
from models.bounding_box import BoundingBox
from numpy import ndarray

from models.prediction import Prediction
from services.faces.yolo_model import YoloModel

class FaceDetector(YoloModel):
    def __init__(self, model_path: str = "weights/yolov8_detection_model.pt") -> None:
        super().__init__(model_path)

    def warm_up(self) -> None:
        self.detect("resources/kad1.jpg")
        print("Face detector warmed up.")

    def detect(self, img: ndarray) -> list[Prediction]:
        results = self.model.predict(img, conf=0.5,show=False, save=False, verbose=False)
        predictions = []
        for box_data in results[0].boxes.data.tolist():
            prediction = Prediction(BoundingBox(Point(box_data[0], box_data[1]), Point(box_data[2], box_data[3])), box_data[4])
            predictions.append(prediction)
        return predictions