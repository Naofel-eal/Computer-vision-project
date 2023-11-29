from DTOs.person_dto import PersonDTO
from mappers.person_mapper import PersonMapper
from models.comparison import Comparison
from models.face import Face
from models.person import Person
from models.prediction import Prediction
from services.face_detector import FaceDetector
from services.face_comparator import FaceComparator
from services.image_editor import ImageEditor
from numpy import ndarray

class PersonManager():
    def __init__(self) -> None:
        self.face_detector: FaceDetector = FaceDetector()
        self.persons: list[Person] = []
    
    def analyze(self, frame_index: int, frame: ndarray) -> None:
        predictions: list[Prediction] = self.face_detector.detect(frame)
        for prediction in predictions:
            face: Face = Face(frame_index, prediction)
            self._save_face(frame, face)
    
    def _save_face(self, frame: ndarray, face: Face) -> None:
        cropped_face: ndarray = ImageEditor.crop(frame, face.prediction.bounding_box)
        if self._is_persons_empty():
            self.persons.append(Person(0, face, cropped_face, face.prediction.confidence))
        else:
            for person in self.persons:
                comparison: Comparison = self.compare(cropped_face, person.cropped_face)
                if comparison.is_same_person:
                    face.distance = comparison.distance
                    if self._is_confidence_higher(person, face):
                        person.cropped_face = ImageEditor.crop(frame, face.prediction.bounding_box)
                        person.cropped_face_confidence = face.prediction.confidence
                    person.faces.append(face)
                    return
            self.persons.append(Person(len(self.persons), face, cropped_face, face.prediction.confidence))

    def _is_confidence_higher(self, person: Person, face: Face) -> bool:
        return person.cropped_face_confidence < face.prediction.confidence

    def compare(self, target_face: ndarray, known_face: ndarray) -> Comparison:
        return FaceComparator.compare(known_face, target_face)

    def _is_persons_empty(self) -> bool:
        return len(self.persons) == 0

    def get_persons(self):
        output_persons: list[PersonDTO] = []
        for person in self.persons:
            output_persons.append(PersonMapper.to_dto(person))
        return output_persons