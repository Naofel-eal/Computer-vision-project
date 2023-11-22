from models.medias.media import Media
from cv2 import VideoCapture, CAP_PROP_FPS
from moviepy.editor import VideoFileClip, ImageSequenceClip

class Video(Media):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.frames = self.get_frames()
        self.audio = self.get_audio()

    def get_frames(self) -> list:
        video = VideoCapture(self.file_path)
        self.fps: int = video.get(CAP_PROP_FPS)
        frames: list = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        return frames

    def get_audio(self) -> VideoFileClip.audio:
        return VideoFileClip(self.file_path).audio

    def merge(self, frames: list) -> ImageSequenceClip:
        render = ImageSequenceClip(frames, fps=self.fps)
        return render