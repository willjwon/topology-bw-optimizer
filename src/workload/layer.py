from src.workload.layer_entry import LayerEntry


class Layer:
    def __init__(self,
                 name: str,
                 forward: LayerEntry,
                 input_grad: LayerEntry,
                 weight_grad: LayerEntry):
        self.name = name
        self.forward = forward
        self.input_grad = input_grad
        self.weight_grad = weight_grad

    def print(self) -> None:
        print(f"{self.name}:")
        print("Forward: ", end='\t\t')
        self.forward.print()
        print("InputGrad: ", end='\t\t')
        self.input_grad.print()
        print("WeightGrad: ", end='\t')
        self.weight_grad.print()
        print()
