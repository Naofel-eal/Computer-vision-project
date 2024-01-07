from cv2 import resize
import numpy as np
import logging
import torch

from models.comparison import Comparison
from services.faces.comparators.face_comparator import FaceComparator
from services.faces.model import Model

class VGGFaceComparator(Model, FaceComparator):
    def __init__(self, model_path: str = "weights/vgg-face.pt"):
        Model.__init__(self, model_path=model_path)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"PyTorch VGGFaceComparator model running on: {self.device}")
        self.model = torch.load(self.model_path).to(self.device)
        self.model.eval()
        self.threshold = 0.40
        self.warm_up()

    def warm_up(self) -> None:
        empty_image = np.ones((224, 224, 3))
        empty_image = self.image_preprocess(empty_image)
        self.model(empty_image)
        logging.info("PyTorch VGGFaceComparator warmed up.")
        
    def compare(self, target_face: np.ndarray,  known_face: np.ndarray, known_face_is_feature: bool = False) -> Comparison:
        target_face_features = self.get_features(target_face)
        if known_face_is_feature:
            known_face_features = known_face
        else:
            known_face_features = self.get_features(known_face)

        return self.compare_features(target_face_features, known_face_features)
    
    def compare_features(self, target_face_features: np.ndarray, known_face_features: np.ndarray) -> Comparison:
        distance: float = self._findCosineDistance(known_face_features, target_face_features)
        is_same_person: bool = self._is_same_person(distance)
        return Comparison(is_same_person, distance)

    def get_features(self, face: np.ndarray) -> np.ndarray:
        face = self.image_preprocess(face)
        with torch.no_grad():
            face_features = self.model(face)
            face_features = face_features[0].cpu().numpy()
            return face_features        

    def image_preprocess(self, img: np.ndarray) -> torch.Tensor:
        image = resize(img, (224, 224))
        image = np.array(image).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(self.device)
        return image
    
    def _findCosineDistance(self, known_face_features: np.ndarray, target_face_features: np.ndarray) -> float:
        a = np.matmul(np.transpose(known_face_features), target_face_features)
        b = np.sum(np.multiply(known_face_features, known_face_features))
        c = np.sum(np.multiply(target_face_features, target_face_features))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def _is_same_person(self, distance: float) -> bool:
        return distance <= self.threshold