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
        bounding_box_sizes = bounding_box.upper_left - bounding_box.bottom_right
        width = bounding_box_sizes[0]
        height = bounding_box_sizes[1]
        y1 = max(0, int(bounding_box.upper_left.y - height * 0.15))
        y2 = min(frame.shape[0], int(bounding_box.bottom_right.y + height * 0.15))
        x1 = max(0, int(bounding_box.upper_left.x - width * 0.15))
        x2 = min(frame.shape[1], int(bounding_box.bottom_right.x + width * 0.15))
        cropped_image = ascontiguousarray(frame[y1:y2, x1:x2])
        blur_cropped_image = cv2.GaussianBlur(cropped_image, (blur_lvl, blur_lvl), 0)
        frame[y1:y2, x1:x2] = blur_cropped_image
        return frame