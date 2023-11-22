from DTOs.person_dto import PersonDTO
from mappers.person_mapper import PersonMapper
from models.face import Face
from models.person import Person
from services.face_detector import FaceDetector
from numpy import ndarray
from services.image_editor import ImageEditor
from face_recognition import compare_faces, face_encodings

class PersonManager():
    def __init__(self) -> None:
        self.face_detector: FaceDetector = FaceDetector()
        self.persons: list[Person] = []
    
    def analyze(self, frame_index: int, frame: ndarray) -> None:
        bounding_boxes: list = self.face_detector.detect(frame)
        for bounding_box in bounding_boxes:
            face: Face = Face(frame_index, bounding_box)
            self.save_face(frame, face)
    
    def save_face(self, frame: ndarray, face: Face) -> None:
        cropped_face: ndarray = ImageEditor.crop(frame, face.bounding_box)
        if len(self.persons) == 0:
            self.persons.append(Person(0, face, cropped_face))
        else:
            for person in self.persons:
                if self.is_same_person(cropped_face, person.cropped_face):
                    person.faces.append(face)
                    return
            self.persons.append(Person(len(self.persons), face, cropped_face))
    
    def get_persons(self):
        output_persons: list[PersonDTO] = []
        for person in self.persons:
            output_persons.append(PersonMapper.to_dto(person))
        return output_persons
    
    def is_same_person(self, target_face: ndarray, known_face: ndarray) -> bool:
        target_face_encoding: ndarray = face_encodings(target_face)[0]
        known_face_encoding: ndarray = face_encodings(known_face)[0]
        return compare_faces([known_face_encoding], target_face_encoding)[0]