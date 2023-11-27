class Point:
    
    def __init__(self, x: int, y: int):
        self.x: int = int(x)
        self.y: int = int(y)
        
    def __sub__(self, P2: 'Point') -> [float, float]:
        width = abs(self.x - P2.x)
        height = abs(self.y - P2.y)
        return (width, height)
