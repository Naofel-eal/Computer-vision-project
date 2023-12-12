from models.medias.media import Media
from cv2 import VideoCapture, CAP_PROP_FPS, COLOR_BGR2RGB, CAP_PROP_POS_FRAMES
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
from numpy import ndarray

from services.images.image_editor import ImageEditor

class Video(Media):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.video = VideoCapture(self.file_path)
        self.fps: int = self.video.get(CAP_PROP_FPS)
        self.audio = self._extract_audio()
        self.current_frame_index = -1

    def _extract_audio(self) -> AudioFileClip:
        video = VideoFileClip(self.file_path)
        return video.audio

    def merge(self, frames: list) -> ImageSequenceClip:
        render = ImageSequenceClip(frames, fps=self.fps)
        render.set_audio(self.audio)
        return render

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