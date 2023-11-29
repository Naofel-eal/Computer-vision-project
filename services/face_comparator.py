from numpy import ndarray
from deepface import DeepFace
from models.comparison import Comparison

class FaceComparator:
    @staticmethod
    def compare(known_face: ndarray, target_face: ndarray) -> Comparison:
        result: dict = DeepFace.verify(known_face, target_face, distance_metric='cosine', enforce_detection=False)
        is_same_person: bool = result.get('verified')
        confidence: float = result.get('distance')
        comparison: Comparison = Comparison(is_same_person, confidence)
        return comparison