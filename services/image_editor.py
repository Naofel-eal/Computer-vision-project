from models.bounding_box import BoundingBox
from numpy import ndarray, ascontiguousarray, interp, array
from PIL import Image, ImageFilter, ImageDraw

class ImageEditor:
    
    @staticmethod
    def crop(frame: ndarray, bounding_box: BoundingBox) -> ndarray:
        y1 = bounding_box.upper_left.y #- 50
        y2 = bounding_box.bottom_right.y #+ 50
        x1 = bounding_box.upper_left.x #- 50
        x2 = bounding_box.bottom_right.x #+ 50
        result = ascontiguousarray(frame[y1:y2, x1:x2])
        return result
    
    @staticmethod
    def blur(frame: ndarray, bounding_box: BoundingBox, blur_lvl: int = 3) -> ndarray:
        image = Image.fromarray(frame)

        bounding_box_sizes = bounding_box.upper_left - bounding_box.bottom_right
        width = bounding_box_sizes[0]
        height = bounding_box_sizes[1]
        y1 = max(0, int(bounding_box.upper_left.y - height * 0.25))
        y2 = min(frame.shape[0], int(bounding_box.bottom_right.y + height * 0.25))
        x1 = max(0, int(bounding_box.upper_left.x - width * 0.25))
        x2 = min(frame.shape[1], int(bounding_box.bottom_right.x + width * 0.25))

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        radius = min((x2 - x1) // 2, (y2 - y1) // 2)

        for r in range(radius, 0, -1):
            mask = Image.new('L', (frame.shape[1], frame.shape[0]), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([(center_x - r, center_y - r), (center_x + r, center_y + r)], fill=255)

            
            blur_amount = interp(r, (radius*0.65, radius), (blur_lvl,0))
            blurred_region = image.filter(ImageFilter.GaussianBlur(blur_amount))
            image.paste(blurred_region, mask=mask)

        blurred_frame = array(image)
        return blurred_frame