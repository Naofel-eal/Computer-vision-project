from datetime import datetime, timedelta

class PerformanceCounter:
    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.elapsed_time: timedelta = None

    def start(self, title: str = "Operation:") -> None:
        self._initialize()
        self.title: str = title
        self.start_time = datetime.now()

    def stop(self) -> None:
        self.end_time = datetime.now()
        self.elapsed_time  = self.end_time - self.start_time
        print(f"{self.title}: {self.elapsed_time.seconds} seconds - {self.elapsed_time.microseconds} microseconds")