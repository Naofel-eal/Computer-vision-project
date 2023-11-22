from models.bounding_box import BoundingBox

class Face:
    
    def __init__(self, frame_index: int, bounding_box: BoundingBox):
        self.frame_index: int = frame_index
        self.bounding_box: BoundingBox = bounding_box
    