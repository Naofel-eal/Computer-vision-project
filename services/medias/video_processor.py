from DTOs.person_dto import PersonDTO
from models.comparison import Comparison
from models.medias.video import Video
from services.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip

class VideoProcessor(MediaProcessor):
    def __init__(self, video: Video) -> None:
        super().__init__()
        self.video = video

    def get_persons(self) -> list[PersonDTO]:
        for index, frame in enumerate(self.video.frames):
            print(f"Processing frame {index}/{len(self.video.frames)}...")
            self.person_manager.analyze(index, frame)
        self.correction()
        return self.person_manager.get_persons()

    def blur_faces(self, personsDTO: list[PersonDTO]) -> ImageSequenceClip:
        for personDTO in personsDTO:
            if personDTO.should_be_blur:
                person = self.person_manager.persons[personDTO.id]
                for face in person.faces:
                    self.video.frames[face.frame_index] = ImageEditor.blur(self.video.frames[face.frame_index], face.prediction.bounding_box)
        return self.video.merge(self.video.frames)
    
    def correction(self) -> None:
        print("Correction")
        for current_person in self.person_manager.persons:
            for current_face in current_person.faces:
                print(f"Correction face {current_face.frame_index}/{len(self.video.frames)} for person {current_person.id + 1}/{len(self.person_manager.persons)}")
                for other_person in self.person_manager.persons:
                    if current_person != other_person:
                        current_cropped_face = ImageEditor.crop(self.video.frames[current_face.frame_index], current_face.prediction.bounding_box)
                        comparison: Comparison = self.person_manager.compare(current_cropped_face, other_person.cropped_face)
                        current_face_distance = current_face.distance
                        other_person_distance = comparison.distance
                        if comparison.is_same_person and other_person_distance < current_face_distance:
                            fig = plt.figure()
                            plt.suptitle(f"Correction: {current_person.id + 1}/{len(self.person_manager.persons)} -> {other_person.id + 1}/{len(self.person_manager.persons)}")
                            ax1 = fig.add_subplot(1, 3, 1)
                            ax2 = fig.add_subplot(1, 3, 2)
                            ax3 = fig.add_subplot(1, 3, 3)
                            ax1.imshow(current_cropped_face)
                            ax2.imshow(current_person.cropped_face)
                            ax3.imshow(other_person.cropped_face)
                            ax1.set_title('Detected face')
                            ax2.set_title(f'Old person: {current_face_distance:.2f}')
                            ax3.set_title(f'New person: {other_person_distance:.2f}')
                            plt.axis('off')
                            plt.show()
                            other_person.faces.append(current_face)
                            current_person.faces.remove(current_face)
                            break