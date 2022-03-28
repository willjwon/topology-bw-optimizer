from abc import *
import torch
from src.workload.workload_parser import WorkloadParser
from src.network.network_parser import NetworkParser
from src.helper.typing import *


class Model:
    def __init__(self,
                 workload_path: str,
                 network_path: str,
                 mp_dim: Optional[List[int]],
                 dp_dim: Optional[List[int]]):
        # network and workload
        workload_parser = WorkloadParser(path=workload_path)
        network_parser = NetworkParser(path=network_path)

        self.workload = workload_parser.parse()
        self.network = network_parser.parse()

        # dims
        self.mp_dim = mp_dim
        self.dp_dim = dp_dim

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
        last_bandwidth = self.network.total_bandwidth - torch.sum(self.torch_bandwidths)
        self.bandwidths = torch.cat((self.torch_bandwidths, last_bandwidth.view(1)))

    def _collective(self,
                    collective_type: Collective,
                    processing_dims: List[int],
                    collective_size: float) -> torch.float64:
        if collective_type == Collective.NoComm:
            return 0.0

        if processing_dims is None:
            return 0.0

        if collective_type == Collective.ReduceScatter:
            messages = list()  # todo: implement message size logic
            assert False, "Not implemented yet"
            exit(-1)

        if collective_type == Collective.AllGather:
            dims_count = len(processing_dims)

            message_size = [collective_size]
            for i in range(dims_count - 1, 0, -1):
                dim = processing_dims[i]
                npus_count = self.network.npus_count[dim]
                new_message_size = message_size[0] * npus_count
                message_size = [new_message_size] + message_size

            messages = list()
            for i in range(dims_count):
                dim = processing_dims[i]
                npus_count = self.network.npus_count[dim]

                new_message_size = message_size[i] * (npus_count - 1)
                messages.append(new_message_size)

        if collective_type == Collective.AllReduce:
            dims_count = len(processing_dims)

            # calculate message size, per NPU
            message_size = [collective_size]
            for i in range(1, dims_count):
                last_dim = processing_dims[i - 1]
                npus_count = self.network.npus_count[last_dim]
                new_message_size = message_size[-1] / npus_count
                message_size.append(new_message_size)

            # calculate message size, at network level
            messages = list()
            for i in range(dims_count):
                dim = processing_dims[i]
                npus_count = self.network.npus_count[dim]

                new_message_size = (message_size[i] / npus_count) * (npus_count - 1) * 2
                messages.append(new_message_size)

        if collective_type == Collective.AllToAll:
            messages = list()  # todo: implement message size logic
            assert False, "Not implemented yet"
            exit(-1)

        # get processing time of each dim
        times = list()
        for i in range(dims_count):
            dim = processing_dims[i]
            time = messages[i] / self.bandwidths[dim]
            times.append(time)

        return max(times)
