from models.bounding_box import BoundingBox
from models.face import Face
from numpy import ndarray

class Person:
    
    def __init__(self, id: int, face: Face, cropped_face: ndarray):
        self.id: int = id
        self.faces: list[Face] = [face]
        self.cropped_face: ndarray = cropped_face
        
    def get_frames_index(self) -> list[int]:
        return [face.frame_index for face in self.faces]
    
    def get_bounding_boxes(self) -> list[BoundingBox]:
        return [face.bounding_box for face in self.faces]
    