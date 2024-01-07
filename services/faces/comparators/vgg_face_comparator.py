from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
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
        empty_image = np.zeros((224, 224, 3))
        empty_image = self.image_preprocess(empty_image)
        self.model(empty_image)
        logging.info("PyTorch VGGFaceComparator warmed up.")
        
    def compare(self, known_face: np.ndarray, target_face: np.ndarray) -> Comparison:
        with torch.no_grad():
            known_face = self.image_preprocess(known_face)
            target_face = self.image_preprocess(target_face)
            known_face_features = self.model(known_face)
            target_face_features = self.model(target_face)
            known_face_features = known_face_features[0].cpu().numpy()
            target_face_features = target_face_features[0].cpu().numpy()

            distance: float = self._findCosineDistance(known_face_features, target_face_features)
            is_same_person: bool = self._is_same_person(distance=distance)
            return Comparison(is_same_person, distance)
        
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