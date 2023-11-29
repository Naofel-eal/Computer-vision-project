from models.bounding_box import BoundingBox
from models.face import Face
from numpy import ndarray

class Person:
    
    def __init__(self, id: int, face: Face, cropped_face: ndarray, cropped_face_confidence: float):
        self.id: int = id
        self.faces: list[Face] = [face]
        self.cropped_face: ndarray = cropped_face
        self.cropped_face_confidence: float = cropped_face_confidence
        
    def get_frames_index(self) -> list[int]:
        return [face.frame_index for face in self.faces]
    
    def get_bounding_boxes(self) -> list[BoundingBox]:
        return [face.prediction.bounding_box for face in self.faces]
    
    def __str__(self) -> str:
        return f"Person n°{self.id} has {len(self.faces)} faces"