from DTOs.person_dto import PersonDTO
from models.medias.media import Media
from services.face_comparator import FaceComparator
from services.person_manager import PersonManager

class MediaProcessor:
    def __init__(self):
        self.person_manager = PersonManager()

    def get_persons(self) -> list[PersonDTO]:
        pass

    def blur_faces(self) -> Media:
        pass