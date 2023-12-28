from DTOs.person_dto import PersonDTO
from models.comparison import Comparison
from models.medias.video import Video
from services.images.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor
from cv2 import VideoWriter, VideoWriter_fourcc
from moviepy.editor import VideoFileClip
from uuid import uuid4
from os import getcwd

class VideoProcessor(MediaProcessor):
    def __init__(self, comparator="VGG-Face") -> None:
        super().__init__(comparator)

    def get_persons(self, video: Video) -> list[PersonDTO]:
        self._analyze(video)
        self._correction(video)
        return self.person_manager.get_persons()

    def _analyze(self, video: Video) -> None:
        index = 0
        frame = video.get_next_frame()
        while frame is not None:
            print(f"Processing frame {index}...")
            self.person_manager.analyze_frame(index, frame)
            frame = video.get_next_frame()
            index += 1

    def _correction(self, video: Video) -> None:
        print("Correction")
        self.person_manager.group_identical_persons()

        for current_person_index, current_person in enumerate(self.person_manager.persons):
            for current_face in current_person.faces:
                for other_person_index, other_person in enumerate(self.person_manager.persons):
                    if current_person_index != other_person_index:
                        current_cropped_face = ImageEditor.crop(video.get_nth_frame(current_face.frame_index), current_face.prediction.bounding_box)
                        comparison: Comparison = self.person_manager.compare_faces(current_cropped_face, other_person.cropped_face)
                        other_person_distance = comparison.distance
                        if comparison.is_same_person:
                            current_face_distance: float = self.person_manager.compare_faces(current_cropped_face, current_person.cropped_face).distance
                            if other_person_distance < current_face_distance:
                                other_person.add_face(current_face)
                                current_person.remove_face(current_face)
                                break

    def save(self, video: Video, personsDTO: list[PersonDTO], output_video_path: str = "results/output.mp4", gradual: bool = False) -> str:
        temp_path = "results/temp.mp4" 
        print("Saving video...")
        fourcc = VideoWriter_fourcc(*'mp4v')
        frame_index = 0
        frame = video.get_nth_frame(frame_index)
        shape = frame.shape
        out = VideoWriter(temp_path, fourcc, video.fps, (shape[1], shape[0]))

        persons_id_to_blur: list[int] = []
        for personDTO in personsDTO:
            if personDTO.should_be_blurred:
                persons_id_to_blur.append(personDTO.id)

        while frame is not None:
            print(f"Saving frame {frame_index}...")
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
            frame = video.get_next_frame()
            frame_index += 1

        out.release()

        video_clip = VideoFileClip(temp_path)
        video_clip: VideoFileClip = video_clip.set_audio(video.audio)
        video_clip.write_videofile(output_video_path, fps=video.fps, codec="libx264", audio_codec="aac")

        return getcwd() + '/' + output_video_path