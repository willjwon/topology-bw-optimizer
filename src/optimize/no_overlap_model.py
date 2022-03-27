import torch
from src.optimize.model import Model


class NoOverlapModel(Model):
    def __init__(self,
                 workload_path: str,
                 network_path: str):
        super().__init__(workload_path, network_path)

    def training_time(self) -> torch.tensor:
        self._sync_bandwidth()

        # calculate time
        training_time = torch.tensor(0, dtype=torch.float64)

        # forward pass
        for layer in self.workload.layers:
            training_time += layer.forward.compute_type

        # backprop
        for i in range(len(self.workload.layers) - 1, -1, -1):
            layer = self.workload.layers[i]
            # layer.print()

        training_time += 100000 / self.bandwidths[-1]

        return training_time
