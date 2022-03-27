from src.helper.typing import *
from src.workload.layer import Layer


class Workload:
    def __init__(self,
                 header: str,
                 layers_count: int,
                 layers: List[Layer]):
        self.header = header
        self.layers_count = layers_count
        self.layers = layers

    def print(self) -> None:
        print("=======================")
        print(self.header)
        print(f"{self.layers_count} layers")
        print("Layer info:")
        for layer in self.layers:
            layer.print()
        print("=======================\n")