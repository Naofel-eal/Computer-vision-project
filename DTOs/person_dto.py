from numpy import ndarray

class PersonDTO:
    def __init__(self, id: int, cropped_face: ndarray, should_be_blur: bool):
        self.id = id
        self.face = cropped_face
        self.should_be_blur = should_be_blur

    def __str__(self) -> str:
        return f"PersonDTO: {self.id} - Should be blur: {self.should_be_blur}"