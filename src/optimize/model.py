from abc import *

import torch

from src.workload.workload_parser import WorkloadParser
from src.network.network_parser import NetworkParser


class Model:
    def __init__(self,
                 workload_path: str,
                 network_path: str):
        # network and workload
        workload_parser = WorkloadParser(path=workload_path)
        network_parser = NetworkParser(path=network_path)

        self.workload = workload_parser.parse()
        self.network = network_parser.parse()

        # torch bandwidth values
        initial_bandwidths = [self.network.total_bandwidth / self.network.dims_count] * (self.network.dims_count - 1)
        self.torch_bandwidths = torch.tensor(data=initial_bandwidths,
                                             dtype=torch.float64,
                                             requires_grad=True)
        self.bandwidths = None
        self._sync_bandwidth()

    @abstractmethod
    def training_time(self) -> torch.tensor:
        self._sync_bandwidth()
        return

    def print_bandwidth(self):
        for bw in self.bandwidths:
            print(f"{bw:.2f}", end=" ")

    def _sync_bandwidth(self) -> None:
        clipped_bw = torch.clip(self.torch_bandwidths, min=0)
        last_bandwidth = self.network.total_bandwidth - torch.sum(clipped_bw)
        self.bandwidths = torch.cat((self.torch_bandwidths, last_bandwidth.view(1)))
