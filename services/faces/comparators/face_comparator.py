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
    def compare(self, known_face: np.ndarray, target_face: np.ndarray) -> Comparison:
        pass