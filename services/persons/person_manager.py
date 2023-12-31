from numpy import ndarray
from uuid import uuid4
import gradio as gr

from mappers.person_mapper import PersonMapper
from models.comparison import Comparison
from models.face import Face
from models.person import Person
from models.prediction import Prediction
from services.faces.comparators.vgg_face_comparator import VGGFaceComparator
from services.faces.face_detector import FaceDetector
from services.faces.comparators.face_comparator import FaceComparator
from services.images.image_editor import ImageEditor

class PersonManager:
    def __init__(self) -> None:
        self.face_detector: FaceDetector = FaceDetector()
        self.face_comparator: FaceComparator = VGGFaceComparator()
        self.persons: list[Person] = []
        
    def reset(self):
        self.persons = []
    
    def analyze_frame(self, frame_index: int, frame: ndarray) -> None:
        persons_id_in_current_frame: list[uuid4] = []
        predictions: list[Prediction] = self.face_detector.detect(frame)
        for prediction in predictions:
            face: Face = Face(frame_index, prediction)
            person_id = self._save_face(frame, face, persons_id_in_current_frame)
            persons_id_in_current_frame.append(person_id)
    
    def _save_face(self, frame: ndarray, face: Face, persons_id_in_current_frame: list[uuid4]) -> uuid4:
            cropped_face: ndarray = ImageEditor.crop(frame, face.prediction.bounding_box)
            cropped_face_features: ndarray = self.face_comparator.get_features(cropped_face)
            if self._is_persons_empty():
                return self._add_person(face, cropped_face, face.prediction.confidence, cropped_face_features)
            else:
                for person in self.persons:
                    if not person.id in persons_id_in_current_frame:
                        comparison: Comparison = self.compare_features(cropped_face_features, person.cropped_face_features)
                        if comparison.is_same_person:
                            if self._is_confidence_higher(person, face):
                                person.replace_cropped_face(cropped_face, face.prediction.confidence, cropped_face_features)
                            person.add_face(face)
                            return person.id
                return self._add_person(face, cropped_face, face.prediction.confidence, cropped_face_features)

    def _add_person(self, face: Face, cropped_face: ndarray, cropped_face_confidence: float, cropped_face_features: ndarray) -> uuid4:
        new_person = Person(face, cropped_face, cropped_face_confidence, cropped_face_features)
        self.persons.append(new_person)
        return new_person.id

    def group_identical_persons(self) -> None:
        gr.Info("Grouping identical persons...")
        for current_person_index, current_person in enumerate(self.persons):
            for other_person_index, other_person in enumerate(self.persons):
                if current_person_index != other_person_index:
                    comparison: Comparison = self.compare_faces(current_person.cropped_face, other_person.cropped_face)
                    if comparison.is_same_person:
                        gr.Info(f"Grouping persons {current_person_index + 1}/{len(self.persons)} and {other_person_index + 1}/{len(self.persons)}")
                        self._merge_persons(current_person, other_person)
    
    def _merge_persons(self, current_person: Person, other_person: Person) -> None:
        person_with_better_confidence = current_person if current_person.cropped_face_confidence > other_person.cropped_face_confidence else other_person
        person_with_worst_confidence = current_person if current_person.cropped_face_confidence < other_person.cropped_face_confidence else other_person
        person_with_better_confidence.add_faces(person_with_worst_confidence.faces)
        self.persons.remove(person_with_worst_confidence)

    def _is_confidence_higher(self, person: Person, face: Face) -> bool:
        return person.cropped_face_confidence < face.prediction.confidence

    def compare_faces(self, target_face: ndarray, known_face: ndarray, known_face_is_feature: bool = False) -> Comparison:
        return self.face_comparator.compare(known_face, target_face, known_face_is_feature)
    
    def compare_features(self, target_face_features: ndarray, known_face_features: ndarray) -> Comparison:
        return self.face_comparator.compare_features(known_face_features, target_face_features)

    def _is_persons_empty(self) -> bool:
        return len(self.persons) == 0

    def get_persons(self):
        return [PersonMapper.to_dto(person) for person in self.persons]

    def get_persons_id_in_frame(self, frame_index: int) -> list[uuid4]:
        output_persons_index: list[uuid4] = []
        for person in self.persons:
            person_frames_index: list[int] = person.get_frames_indexes()
            if frame_index in person_frames_index:
                output_persons_index.append(person.id)
        return output_persons_index