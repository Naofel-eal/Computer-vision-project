from DTOs.person_dto import PersonDTO
from models.medias.media import Media
from services.persons.person_manager import PersonManager

class MediaProcessor:
    def __init__(self):
        self.person_manager = PersonManager()
        
    def reset(self):
        self.person_manager.reset()

    def get_persons(self) -> list[PersonDTO]:
        pass

    def blur_faces(self) -> Media:
        pass