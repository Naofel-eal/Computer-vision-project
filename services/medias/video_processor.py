from cv2 import VideoWriter, VideoWriter_fourcc
from moviepy.editor import VideoFileClip
from uuid import uuid4
from os import remove, path
import gradio as gr
import logging

from DTOs.person_dto import PersonDTO
from models.comparison import Comparison
from models.medias.video import Video
from services.images.image_editor import ImageEditor
from services.medias.media_processor import MediaProcessor

class VideoProcessor(MediaProcessor):
    def __init__(self) -> None:
        super().__init__()

    def get_persons(self, video: Video) -> list[PersonDTO]:
        self._analyze(video)
        self._correction(video)
        return self.person_manager.get_persons()

    def _analyze(self, video: Video, progress=gr.Progress()) -> None:
        gr.Info("Analysis in progress...")
        progress(0)
        for i in progress.tqdm(range(video.frame_count), desc="Analyzing video", total=video.frame_count):
            frame = video.get_nth_frame(i)
            if frame is not None:
                self.person_manager.analyze_frame(i, frame)
            else:
                logging.warning(f"Frame {i} not found")
                break

    def _correction(self, video: Video, progress=gr.Progress()) -> None:
        gr.Info("Correction in progress...")
        self.person_manager.group_identical_persons()
        total_faces_count = sum([len(person.faces) for person in self.person_manager.persons])

        for current_person_index, current_person in progress.tqdm(enumerate(self.person_manager.persons), desc="Correcting faces", total=total_faces_count):
            for current_face in current_person.faces:
                current_croped_face = ImageEditor.crop(video.get_nth_frame(current_face.frame_index), current_face.prediction.bounding_box)
                current_cropped_face_features = self.person_manager.face_comparator.get_features(current_croped_face)
                for other_person_index, other_person in enumerate(self.person_manager.persons):
                    if current_person_index != other_person_index:
                        other_person_face_features = other_person.cropped_face_features
                        comparison: Comparison = self.person_manager.compare_features(current_cropped_face_features, other_person_face_features)
                        other_person_distance = comparison.distance
                        if comparison.is_same_person:
                            current_person_distance: float = self.person_manager.compare_features(current_cropped_face_features, current_person.cropped_face_features).distance
                            if other_person_distance < current_person_distance:
                                other_person.add_face(current_face)
                                current_person.remove_face(current_face)
                                break

    def save(self, video: Video, personsDTO: list[PersonDTO], output_video_path: str = "results/output.mp4", gradual: bool = False, progress=gr.Progress()) -> str:
        gr.Info("Applying blur...")
        progress(0)
        temp_path = "results/temp.mp4" 
        fourcc = VideoWriter_fourcc(*'mp4v')
        frame_index = 0
        frame = video.get_nth_frame(frame_index)
        shape = frame.shape
        out = VideoWriter(temp_path, fourcc, video.fps, (shape[1], shape[0]))

        persons_id_to_blur: list[int] = []
        for personDTO in personsDTO:
            if personDTO.should_be_blurred:
                persons_id_to_blur.append(personDTO.id)
            
        frames_index_to_blur = set()
        for person_id in persons_id_to_blur:
            for person in self.person_manager.persons:
                if person.id == person_id:
                    frames_index_to_blur.update(person.get_frames_indexes())
                    break

        for frame_index in progress.tqdm(range(video.frame_count), desc="Saving video", total=video.frame_count):
            frame = video.get_nth_frame(frame_index)
            if frame is None:
                break

            if frame_index in frames_index_to_blur:
                persons_id_in_current_frame: list[uuid4] = self.person_manager.get_persons_id_in_frame(frame_index)
                for person_id in persons_id_in_current_frame:
                    if person_id in persons_id_to_blur:
                        person = next((person for person in self.person_manager.persons if person.id == person_id), None)
                        if person is not None:
                            frame = ImageEditor.blur(frame, person.get_face(frame_index).prediction.bounding_box, gradual=gradual)
                        else:
                            logging.warning(f"Person with id {person_id} not found")
            
            frame = ImageEditor.RGB_to_BGR(frame)
            out.write(frame)
        out.release()

        video_clip = VideoFileClip(temp_path)
        video_clip: VideoFileClip = video_clip.set_audio(video.audio)
        video_clip.write_videofile(output_video_path, fps=video.fps, codec="libx264", audio_codec="aac")

        remove(temp_path)

        return path.abspath(output_video_path)