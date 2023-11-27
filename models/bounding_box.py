from models.point import Point

class BoundingBox:
    
    def __init__(self, upper_left: Point, bottom_right: Point):
        self.upper_left: Point = upper_left
        self.bottom_right: Point = bottom_right
    
    def __str__(self) -> str:
        return f"({self.upper_left}, {self.bottom_right})"