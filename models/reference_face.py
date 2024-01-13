from numpy import ndarray

class ReferenceFace:
    def __init__(self, cropped_face, cropped_face_confidence, cropped_face_features) -> None:
        self.cropped_face: ndarray = cropped_face
        self.cropped_face_confidence: float = cropped_face_confidence
        self.cropped_face_features: ndarray = cropped_face_features