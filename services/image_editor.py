from models.bounding_box import BoundingBox
from numpy import ndarray, ascontiguousarray
import cv2

class ImageEditor:
    
    @staticmethod
    def crop(frame: ndarray, bounding_box: BoundingBox) -> ndarray:
        y1 = bounding_box.upper_left.y
        y2 = bounding_box.bottom_right.y
        x1 = bounding_box.upper_left.x
        x2 = bounding_box.bottom_right.x
        result = ascontiguousarray(frame[y1:y2, x1:x2])
        return result
    
    @staticmethod
    def blur(frame: ndarray, bounding_box: BoundingBox, blur_lvl: int) -> ndarray:
        # Idéalement mettre zone de flou sup. en % et flou progressif
        y1 = max(0, bounding_box.upper_left.y - 15)
        y2 = min(frame.shape[0], bounding_box.bottom_right.y + 15)
        x1 = max(0, bounding_box.upper_left.x - 15)
        x2 = min(frame.shape[1], bounding_box.bottom_right.x + 15)
        cropped_image = frame[y1:y2, x1:x2]
        blur_cropped_image = cv2.GaussianBlur(cropped_image, (blur_lvl, blur_lvl), 0)
        frame[y1:y2, x1:x2] = blur_cropped_image
        return frame