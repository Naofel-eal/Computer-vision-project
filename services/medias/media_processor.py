from DTOs.person_dto import PersonDTO
from models.medias.media import Media
from services.persons.person_manager import PersonManager

class MediaProcessor:
    def __init__(self, comparator: str = "VGG-Face"):
        self.person_manager = PersonManager(comparator)

    def get_persons(self) -> list[PersonDTO]:
        pass

    def blur_faces(self) -> Media:
        pass