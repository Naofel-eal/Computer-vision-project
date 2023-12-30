from datetime import datetime, timedelta
import logging

class PerformanceCounter:
    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        self.start_time: datetime = None
        self.end_time: datetime = None
        self.elapsed_time: timedelta = None
        self.title: str = None
        self.indentation: str = None
        self.importance: str = None

    def measure(self, title: str = "Operation", importance: str = 3) -> None:
        self._initialize()
        self.title: str = title
        self.indentation: str = " " * (3 - importance)
        self._print_title(importance)
        self.start_time = datetime.now()
    
    def _print_title(self, importance: str) -> None:
        logging.info(f"{self.indentation}{'#' * importance}-{self.title}-{'#' * importance}")

    def stop(self) -> timedelta:
        self.end_time = datetime.now()
        self.elapsed_time  = self.end_time - self.start_time
        print(f"{self.indentation}{self.title}: {self.elapsed_time}")
        return self.elapsed_time
