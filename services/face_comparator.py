from models.comparison import Comparison
import numpy as np
from deepface import DeepFace

from services.face_model import FaceModel

class FaceComparator(FaceModel):
    def __init__(self, model_path: str = "weights/yolov8_classification_model.pt", useDeepFace: bool = False) -> None:
        self.useDeepFace = useDeepFace
        if(self.useDeepFace):
            self.warm_up_deepFace()
        else:
            super().__init__(model_path, "comparator")
            
    def warm_up_deepFace(self) -> None:
        self.compare("resources/kad.jpg", "resources/kad.jpg")
        print("DeepFace comparator warmed up.")
        
    def compare(self, known_face: np.ndarray, target_face: np.ndarray) -> Comparison:
        if(self.useDeepFace):
            result: dict = DeepFace.verify(known_face, target_face, distance_metric='cosine', detector_backend="skip", enforce_detection=False)
            is_same_person: bool = result.get('verified')
            confidence: float = result.get('distance')
        else:
            known_face_features = self.model.predict(known_face, show=False, save=False, verbose=False)
            target_face_features = self.model.predict(target_face, show=False, save=False, verbose=False)
            known_face_features = known_face_features[0].cpu().numpy()
            target_face_features = target_face_features[0].cpu().numpy()
            dist: float = self.findCosineDistance(known_face_features, target_face_features)
            is_same_person: bool = self.is_same_person(distance=dist, threshold=0.30)
            confidence: float = dist
        
        comparison: Comparison = Comparison(is_same_person, confidence)
        return comparison
    
    def findCosineDistance(self, known_face_features, target_face_features) -> float:
        a = np.matmul(np.transpose(known_face_features), target_face_features)
        b = np.sum(np.multiply(known_face_features, known_face_features))
        c = np.sum(np.multiply(target_face_features, target_face_features))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def is_same_person(self, distance, threshold) -> bool:
        return distance <= threshold