from models.bounding_box import BoundingBox
from models.face import Face
from models.reference_face import ReferenceFace
from numpy import ndarray
from uuid import uuid4

class Person:
    
    def __init__(self, face: Face, cropped_face: ndarray, cropped_face_confidence: float, cropped_face_features: ndarray = None):
        self.id = uuid4()
        self.faces: list[Face] = [face]
        self.reference_face: ReferenceFace = ReferenceFace(cropped_face, cropped_face_confidence, cropped_face_features)
        
    def get_frames_indexes(self) -> list[int]:
        return [face.frame_index for face in self.faces]
    
    def get_bounding_boxes(self) -> list[BoundingBox]:
        return [face.prediction.bounding_box for face in self.faces]
    
    def get_face(self, frame_index: int) -> Face:
        for face in self.faces:
            if face.frame_index == frame_index:
                return face
        return None
    
    def add_face(self, face: Face) -> None:
        self.faces.append(face)

    def add_faces(self, faces: list[Face]) -> None:
        self.faces.extend(faces)
    
    def remove_face(self, face: Face) -> None:
        for current_face in self.faces:
            if current_face == face:
                self.faces.remove(current_face)
                break

    def replace_reference_face(self, cropped_face: ndarray, cropped_face_confidence: float, cropped_face_features) -> None:
        self.reference_face = ReferenceFace(cropped_face, cropped_face_confidence, cropped_face_features)

    def __str__(self) -> str:
        return f"Person n°{self.id} has {len(self.faces)} faces"

    def __eq__(self, other: 'Person'):
        if not isinstance(other, Person):
            return NotImplemented
        
        return self.id == other.id and (self.cropped_face == other.cropped_face).all() and self.cropped_face_confidence == other.cropped_face_confidence