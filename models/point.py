class Point:
    
    def __init__(self, x: int, y: int):
        self.x: int = int(x)
        self.y: int = int(y)
        
    def __sub__(self, P2: 'Point') -> [float, float]:
        width = abs(self.x - P2.x)
        height = abs(self.y - P2.y)
        return (width, height)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        
        return self.x == other.x and self.y == other.y
