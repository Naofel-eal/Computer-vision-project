from models.comparison import Comparison
from services.faces.comparators.face_comparator import FaceComparator
from services.faces.yolo_model import YoloModel
import numpy as np

class YoloComparator(YoloModel, FaceComparator):
    def __init__(self, model_path: str = "weights/yolov8_classification_model.pt"):
        YoloModel.__init__(self, model_path=model_path)
        self.change_model_names()
        self.warm_up()
    
    def change_model_names(self):
        for index, name in enumerate(self.model.names):
            self.model.names[index] = f'person{index}'

    def warm_up(self) -> None:
        self.compare("resources/kad1.jpg", "resources/kad1.jpg")
        print("Yolo comparator warmed up.")

    def compare(self, known_face: np.ndarray, target_face: np.ndarray) -> Comparison:
        known_face_features = self.model.predict(known_face, show=False, save=False, verbose=False)
        target_face_features = self.model.predict(target_face, show=False, save=False, verbose=False)
        known_face_features = known_face_features[0].cpu().numpy()
        target_face_features = target_face_features[0].cpu().numpy()

        distance: float = self._findCosineDistance(known_face_features, target_face_features)
        is_same_person: bool = self._is_same_person(distance=distance, threshold=0.4758)
        return Comparison(is_same_person, distance)
    
    def _findCosineDistance(self, known_face_features: np.ndarray, target_face_features: np.ndarray) -> float:
        a = np.matmul(np.transpose(known_face_features), target_face_features)
        b = np.sum(np.multiply(known_face_features, known_face_features))
        c = np.sum(np.multiply(target_face_features, target_face_features))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def _is_same_person(self, distance: float, threshold: float) -> bool:
        return distance <= threshold