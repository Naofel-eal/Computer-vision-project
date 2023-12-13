from models.comparison import Comparison
from services.faces.comparators.face_comparator import FaceComparator
import numpy as np
import torch

from services.faces.model import Model

class VGGFaceComparator(Model, FaceComparator):
    def __init__(self, model_path: str = "weights/vgg-face.pt"):
        Model.__init__(self, model_path=model_path)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch VGGFaceComparator model running on: {self.device}")
        self.model = torch.load(self.model_path).to(self.device)
        self.model.eval()
        self.warm_up()

    def warm_up(self) -> None:
        self.compare("resources/kad1.jpg", "resources/kad1.jpg")
        print("PyTorch VGGFaceComparator warmed up.")

    def compare(self, known_face: np.ndarray, target_face: np.ndarray) -> Comparison:
        with torch.no_grad():
            known_face = torch.from_numpy(known_face).to(self.device)
            target_face = torch.from_numpy(target_face).to(self.device)
            known_face_features = self.model(known_face)
            target_face_features = self.model(target_face)
            known_face_features = known_face_features[0].cpu().numpy()
            target_face_features = target_face_features[0].cpu().numpy()

            distance: float = self._findCosineDistance(known_face_features, target_face_features)
            is_same_person: bool = self._is_same_person(distance=distance, threshold=0.4)
            return Comparison(is_same_person, distance)
    
    def _findCosineDistance(self, known_face_features: np.ndarray, target_face_features: np.ndarray) -> float:
        a = np.matmul(np.transpose(known_face_features), target_face_features)
        b = np.sum(np.multiply(known_face_features, known_face_features))
        c = np.sum(np.multiply(target_face_features, target_face_features))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def _is_same_person(self, distance: float, threshold: float) -> bool:
        return distance <= threshold