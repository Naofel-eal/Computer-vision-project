from models.point import Point

class BoundingBox:
    
    def __init__(self, upper_left: Point, bottom_right: Point):
        self.upper_left: Point = upper_left
        self.bottom_right: Point = bottom_right
    
    def __str__(self) -> str:
        return f"({self.upper_left}, {self.bottom_right})"
    
    def __eq__(self, other: 'BoundingBox'):
        if not isinstance(other, BoundingBox):
            return NotImplemented
        
        return self.upper_left == other.upper_left and self.bottom_right == other.bottom_right