from numpy import ndarray
from models.comparison import Comparison
from deepface import DeepFace
from services.faces.comparators.face_comparator import FaceComparator

class DeepFaceComparator(FaceComparator):
    def __init__(self) -> None:
        self.warm_up()

    def warm_up(self) -> None:
        self.compare("resources/kad1.jpg", "resources/kad1.jpg")
        print("DeepFace comparator warmed up.")

    def compare(self, known_face: ndarray, target_face: ndarray) -> Comparison:
        result: dict = DeepFace.verify(known_face, target_face, distance_metric='cosine', detector_backend="skip", enforce_detection=False)
        is_same_person: bool = result.get('verified')
        distance: float = result.get('distance')
        return Comparison(is_same_person, distance)