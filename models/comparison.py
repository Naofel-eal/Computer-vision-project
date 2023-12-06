class Comparison:
    def __init__(self, is_same_person: bool, distance: float) -> None:
        self.is_same_person = is_same_person
        self.distance = distance
    
    def __str__(self) -> str:
        return f"Is same person: {self.is_same_person}, distance: {self.distance:.2f}"
    
    def __eq__(self, other: 'Comparison') -> bool:
        if not isinstance(other, Comparison):
            return NotImplemented
        
        return self.is_same_person == other.is_same_person and self.distance == other.distance