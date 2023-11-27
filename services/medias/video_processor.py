from DTOs.person_dto import PersonDTO
from models.medias.video import Video
from services.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor

class VideoProcessor(MediaProcessor):
    def __init__(self, video: Video) -> None:
        super().__init__()
        self.video = video

    def get_persons(self) -> list[PersonDTO]:
        for index, frame in enumerate(self.video.frames):
            self.person_manager.analyze(index, frame)
        return self.person_manager.get_persons()
    
    def blur_faces(self, personsDTO: list[PersonDTO]) -> Video:
        for personDTO in personsDTO:
            if personDTO.should_be_blur:
                person = self.person_manager.persons[personDTO.id]
                for face in person.faces:
                    self.video.frames[face.frame_index] = ImageEditor.blur(self.video.frames[face.frame_index], face.bounding_box)
        return self.video.merge(self.video.frames)