from DTOs.person_dto import PersonDTO
from models.medias.image import Image
from numpy import ndarray
from services.images.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor

class ImageProcessor(MediaProcessor):
    def __init__(self, comparator="VGG-Face"):
        super().__init__(comparator)
        
    def reset(self):
        self.person_manager.reset()

    def get_persons(self, frame: ndarray) -> list[PersonDTO]:
        self.person_manager.analyze_frame(0, frame)
        return self.person_manager.get_persons()
    
    def apply_blur(self, frame: ndarray, personsDTO: list[PersonDTO], gradual: bool = False) -> Image:
        persons_id_to_blur: list[int] = []
        for personDTO in personsDTO:
            if personDTO.should_be_blurred:
                persons_id_to_blur.append(personDTO.id)
        
        for person_id in persons_id_to_blur:       
            person = next((person for person in self.person_manager.persons if person.id == person_id), None)
            if person is not None:
                frame = ImageEditor.blur(frame, person.get_face(0).prediction.bounding_box, gradual=gradual)
            else:
                print(f"Person with id {person_id} not found")

        return frame