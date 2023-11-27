from models.medias.media import Media
from cv2 import VideoCapture, CAP_PROP_FPS, cvtColor, COLOR_BGR2RGB
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
from numpy import ndarray

class Video(Media):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.frames: list[ndarray] = self._extract_frames()
        self.audio = self._extract_audio()

    def _extract_frames(self) -> list[ndarray]:
        video = VideoCapture(self.file_path)
        self.fps: int = video.get(CAP_PROP_FPS)
        frames: list[ndarray] = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cvtColor(frame, COLOR_BGR2RGB)
            frames.append(frame)
        video.release()
        return frames

    def _extract_audio(self) -> AudioFileClip:
        video = VideoFileClip(self.file_path)
        return video.audio

    def merge(self, frames: list) -> ImageSequenceClip:
        render = ImageSequenceClip(frames, fps=self.fps)
        render.set_audio(self.audio)
        return render