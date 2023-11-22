from models.medias.media import Media
from numpy import ndarray, asarray
from PIL import Image as Img

class Image(Media):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.image: ndarray =  asarray(Img.open(file_path))