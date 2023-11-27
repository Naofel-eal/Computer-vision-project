class Point:
    
    def __init__(self, x: int, y: int):
        self.x: int = int(x)
        self.y: int = int(y)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    