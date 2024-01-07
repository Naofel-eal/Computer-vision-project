from abc import abstractmethod
from models.comparison import Comparison
import numpy as np

class FaceComparator():
    @abstractmethod 
    def __init__(self) -> None:
        self.warm_up()

    @abstractmethod    
    def warm_up(self) -> None:
        pass

    @abstractmethod
    def compare(self, target_face: np.ndarray, known_face: np.ndarray, known_face_is_feature: bool = False) -> Comparison:
        pass

    @abstractmethod
    def get_features(self, face: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compare_features(self, target_face_features: np.ndarray, known_face_features: np.ndarray) -> Comparison:
        pass