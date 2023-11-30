from DTOs.person_dto import PersonDTO
from models.comparison import Comparison
from models.medias.video import Video
from services.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor
from utils.performance_counter import PerformanceCounter
from matplotlib import pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc

class VideoProcessor(MediaProcessor):
    def __init__(self, video: Video) -> None:
        super().__init__()
        self.video = video
        self.performance_counter = PerformanceCounter()

    def get_persons(self) -> list[PersonDTO]:
        self._analyze()        
        self._correction()
        return self.person_manager.get_persons()

    def _analyze(self) -> None:
        self.performance_counter.start("Analyzing time:")
        index = 0
        frame = self.video.get_next_frame()
        while frame is not None:
            print(f"Processing frame {index}...")
            self.person_manager.analyze(index, frame)
            frame = self.video.get_next_frame()
            index += 1
        self.performance_counter.stop()

    def _correction(self) -> None:
        self.performance_counter.start("Correction time:")
        print("Correction")
        for current_person in self.person_manager.persons:
            for current_face in current_person.faces:
                for other_person in self.person_manager.persons:
                    if current_person != other_person:
                        current_cropped_face = ImageEditor.crop(self.video.get_nth_frame(current_face.frame_index), current_face.prediction.bounding_box)
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
        self.performance_counter.stop()

    def save(self, personsDTO: list[PersonDTO], output_video_path: str = "output.mp4") -> None:
        fourcc = VideoWriter_fourcc(*'MP4V')
        frame_index = 0
        frame = self.video.get_nth_frame(frame_index)
        shape = frame.shape
        out = VideoWriter(output_video_path, fourcc, self.video.fps, (shape[1], shape[0]))

        persons_id_to_blur: list[int] = []
        for personDTO in personsDTO:
            if personDTO.should_be_blur:
                persons_id_to_blur.append(personDTO.id)

        while frame is not None:
            print(f"Saving frame {frame_index}")
            persons_in_this_frame: list[int] = self.person_manager.get_persons_id_in_frame(frame_index)
            for person_id in persons_in_this_frame:
                if person_id in persons_id_to_blur:
                    person = self.person_manager.persons[person_id]
                    frame = ImageEditor.blur(frame, person.get_face(frame_index).prediction.bounding_box)
            
            frame = ImageEditor.RGB_to_BGR(frame)
            out.write(frame)
            frame = self.video.get_next_frame()
            frame_index += 1

        out.release()