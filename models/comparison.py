class Comparison:
    def __init__(self, is_same_person: bool, distance: float) -> None:
        self.is_same_person = is_same_person
        self.distance = distance
    
    def __str__(self) -> str:
        return f"Is same person: {self.is_same_person}, distance: {self.distance:.2f}"