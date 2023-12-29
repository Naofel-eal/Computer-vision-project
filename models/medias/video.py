from models.medias.media import Media
from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, CAP_PROP_FPS, CAP_PROP_POS_FRAMES
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
from numpy import ndarray
import gradio as gr

from services.images.image_editor import ImageEditor

class Video(Media):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.video = VideoCapture(self.file_path)
        self.fps: int = self.video.get(CAP_PROP_FPS)
        self.audio = self._extract_audio()
        self.current_frame_index = -1
        self.frame_count = int(self.video.get(CAP_PROP_FRAME_COUNT))

    def _extract_audio(self) -> AudioFileClip:
        audio = VideoFileClip(self.file_path).audio
        return audio

    def get_next_frame(self) -> ndarray:
        ret, frame = self.video.read()
        if ret:
            self.current_frame_index += 1
            return ImageEditor.BGR_to_RGB(frame)
        return None

    def get_nth_frame(self, n: int) -> ndarray:
        self.video.set(CAP_PROP_POS_FRAMES, n - 1)
        self.current_frame_index = n - 1
        return self.get_next_frame()