from numpy import ndarray
from deepface import DeepFace
from models.comparison import Comparison
from utils.performance_counter import PerformanceCounter

class FaceComparator:
    @staticmethod
    def compare(known_face: ndarray, target_face: ndarray) -> Comparison:
        PerformanceCounter.start("Face comparison time:")
        result: dict = DeepFace.verify(known_face, target_face, distance_metric='cosine', enforce_detection=False)
        PerformanceCounter.stop()
        is_same_person: bool = result.get('verified')
        confidence: float = result.get('distance')
        comparison: Comparison = Comparison(is_same_person, confidence)
        return comparison