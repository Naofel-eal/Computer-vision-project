from DTOs.person_dto import PersonDTO
from models.person import Person
from services.images.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor

class ImageProcessor(MediaProcessor):
    def __init__(self, image):
        super().__init__()
        self.image = image

    def get_persons(self) -> list[PersonDTO]:
        self.person_manager.analyze_frame(0, self.image)
        return self.person_manager.get_persons()
    
    def blur_faces(self, personsDTO: list[PersonDTO]):
        for personDTO in personsDTO:
            if personDTO.should_be_blurred:
                person: Person = self.person_manager.persons[personDTO.id]
                for face in person.faces:
                    self.image = ImageEditor.blur(self.image, face.prediction.bounding_box)
        return self.image