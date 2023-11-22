from models.bounding_box import BoundingBox
from numpy import ndarray

class ImageEditor:
    
    @staticmethod
    def crop(frame: ndarray, bounding_box: BoundingBox) -> ndarray:
        y1 = bounding_box.upper_left.y
        y2 = bounding_box.bottom_right.y
        x1 = bounding_box.upper_left.x
        x2 = bounding_box.bottom_right.x
        return frame[y1:y2, x1:x2]