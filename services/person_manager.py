from datetime import timedelta
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

from utils.performance_counter import PerformanceCounter

class PersonManager():
    def __init__(self) -> None:
        self.face_detector: FaceDetector = FaceDetector()
        self.face_comparator: FaceComparator = FaceComparator()
        self.persons: list[Person] = []
        self.performance_counter = PerformanceCounter()
        self.detection_times: list[timedelta] = []
        self.comparison_times: list[timedelta] = []
    
    def analyze(self, frame_index: int, frame: ndarray) -> None:
        self.performance_counter.measure("Face detection", 1)
        predictions: list[Prediction] = self.face_detector.detect(frame)
        self.detection_times.append(self.performance_counter.stop())
        for prediction in predictions:
            face: Face = Face(frame_index, prediction)
            self._save_face(frame, face)
    
    def _save_face(self, frame: ndarray, face: Face) -> None:
        cropped_face: ndarray = ImageEditor.crop(frame, face.prediction.bounding_box)
        if self._is_persons_empty():
            self.persons.append(Person(0, face, cropped_face, face.prediction.confidence))
        else:
            for person in self.persons:
                self.performance_counter.measure("Face comparison", 1)
                comparison: Comparison = self.compare(cropped_face, person.cropped_face)
                self.comparison_times.append(self.performance_counter.stop())
                if comparison.is_same_person:
                    if self._is_confidence_higher(person, face):
                        person.replace_cropped_face(cropped_face, face.prediction.confidence)
                    person.add_face(face)
                    return
            self.persons.append(Person(len(self.persons), face, cropped_face, face.prediction.confidence))

    def _is_confidence_higher(self, person: Person, face: Face) -> bool:
        return person.cropped_face_confidence < face.prediction.confidence

    def compare(self, target_face: ndarray, known_face: ndarray) -> Comparison:
        return self.face_comparator.compare(known_face, target_face)

    def _is_persons_empty(self) -> bool:
        return len(self.persons) == 0

    def get_persons(self):
        output_persons: list[PersonDTO] = []
        for person in self.persons:
            output_persons.append(PersonMapper.to_dto(person))

        print(f"Detection times average: {sum(self.detection_times, timedelta()) / len(self.detection_times)}")
        print(f"Comparison times average: {sum(self.comparison_times, timedelta()) / len(self.comparison_times)}")
        return output_persons

    def get_persons_id_in_frame(self, frame_index: int) -> list[int]:
        output_persons_id: list[int] = []
        for person in self.persons:
            person_frames_index: list[int] = person.get_frames_indexes()
            if frame_index in person_frames_index:
                output_persons_id.append(person.id)
        return output_persons_id