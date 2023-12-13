from DTOs.person_dto import PersonDTO
from models.comparison import Comparison
from models.medias.video import Video
from services.images.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor
from matplotlib import pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
from uuid import uuid4

class VideoProcessor(MediaProcessor):
    def __init__(self, video: Video, comparator="VGG-Face") -> None:
        super().__init__(comparator)
        self.video = video

    def get_persons(self) -> list[PersonDTO]:
        self._analyze()
        self._correction()
        return self.person_manager.get_persons()

    def _analyze(self) -> None:
        index = 0
        frame = self.video.get_next_frame()
        while frame is not None:
            print(f"Processing frame {index}...")
            self.person_manager.analyze_frame(index, frame)
            frame = self.video.get_next_frame()
            index += 1
        for index, person in enumerate(self.person_manager.persons):
            plt.suptitle("Detected persons during analysis")
            plt.subplot(1, len(self.person_manager.persons), index+1)
            plt.imshow(person.cropped_face)
            plt.axis('off')
        plt.show()

    def _correction(self) -> None:
        print("Correction")
        self.person_manager.group_identical_persons()

        for current_person_index, current_person in enumerate(self.person_manager.persons):
            for current_face in current_person.faces:
                for other_person_index, other_person in enumerate(self.person_manager.persons):
                    if current_person_index != other_person_index:
                        current_cropped_face = ImageEditor.crop(self.video.get_nth_frame(current_face.frame_index), current_face.prediction.bounding_box)
                        comparison: Comparison = self.person_manager.compare_faces(current_cropped_face, other_person.cropped_face)
                        other_person_distance = comparison.distance
                        if comparison.is_same_person:
                            current_face_distance: float = self.person_manager.compare_faces(current_cropped_face, current_person.cropped_face).distance
                            if other_person_distance < current_face_distance:
                                fig = plt.figure()
                                plt.suptitle(f"Frame {current_face.frame_index} - Correction: {current_person_index + 1}/{len(self.person_manager.persons)} -> {other_person_index + 1}/{len(self.person_manager.persons)}")
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
                                other_person.add_face(current_face)
                                current_person.remove_face(current_face)
                                break

    def save(self, personsDTO: list[PersonDTO], output_video_path: str = "results/output.mp4", gradual: bool = False) -> None:
        print("Saving video...")
        fourcc = VideoWriter_fourcc(*'MP4V')
        frame_index = 0
        frame = self.video.get_nth_frame(frame_index)
        shape = frame.shape
        out = VideoWriter(output_video_path, fourcc, self.video.fps, (shape[1], shape[0]))

        persons_id_to_blur: list[int] = []
        for personDTO in personsDTO:
            if personDTO.should_be_blurred:
                persons_id_to_blur.append(personDTO.id)

        while frame is not None:
            print(f"Saving frame {frame_index}")
            persons_id_in_current_frame: list[uuid4] = self.person_manager.get_persons_id_in_frame(frame_index)
            for person_id in persons_id_in_current_frame:
                if person_id in persons_id_to_blur:
                    person = next((person for person in self.person_manager.persons if person.id == person_id), None)
                    if person is not None:
                        frame = ImageEditor.blur(frame, person.get_face(frame_index).prediction.bounding_box, gradual=gradual)
                    else:
                        print(f"Person with id {person_id} not found")
            
            frame = ImageEditor.RGB_to_BGR(frame)
            out.write(frame)
            frame = self.video.get_next_frame()
            frame_index += 1

        out.release()