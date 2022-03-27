from src.helper.typing import *


class LayerEntry:
    def __init__(self,
                 compute_time: float,
                 comm_type: Collective,
                 comm_size: float):
        self.compute_time = compute_time
        self.comm_type = comm_type
        self.comm_size = comm_size

    def print(self) -> None:
        print(f"{self.compute_time}\t{self.comm_type}\t{self.comm_size}")
