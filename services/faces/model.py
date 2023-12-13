from abc import abstractmethod

class Model:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        
    @abstractmethod
    def warm_up(self) -> None:
        pass
