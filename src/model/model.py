from abc import *
import torch
import math
from src.workload.workload_parser import WorkloadParser
from src.network.network import Network
from src.helper.typing import *
from src.cost.cost_calculator import CostCalculator


class Model:
    network = None
    cost_calculator = None
    torch_bandwidths = None
    bandwidths = None

    def __init__(self,
                 workload_path: str,
                 mp_size: int,
                 dp_size: int):
        # workload
        workload_parser = WorkloadParser(path=workload_path)
        self.workload = workload_parser.parse()

        # dims
        self.mp_dim = self._get_mp_dim(mp_size=mp_size)
        self.dp_dim = self._get_dp_dim(dp_size=dp_size)

        self.mp_npus_count = self._get_mp_npus_count(mp_size=mp_size, mp_dim=self.mp_dim)
        self.dp_npus_count = self._get_dp_npus_count(dp_size=dp_size, dp_dim=self.dp_dim)

    @abstractmethod
    def training_time(self) -> torch.tensor:
        return

    @staticmethod
    def init(network: Network,
             cost_calculator: CostCalculator = None) -> None:
        # set class variables
        Model.network = network
        Model.cost_calculator = cost_calculator

        # torch bandwidth values
        initial_bandwidths = [network.total_bandwidth / network.dims_count] * (network.dims_count - 1)
        Model.torch_bandwidths = torch.tensor(data=initial_bandwidths,
                                              dtype=torch.float64,
                                              requires_grad=True)
        Model.sync_bandwidth()

    @staticmethod
    def sync_bandwidth() -> None:
        last_bandwidth = Model.network.total_bandwidth - torch.sum(Model.torch_bandwidths)
        Model.bandwidths = torch.cat((Model.torch_bandwidths, last_bandwidth.view(1)))

    def print_bandwidth(self):
        for bw in Model.bandwidths:
            print(f"{bw:.2f}", end=" ")
        print()

    @staticmethod
    def _get_mp_dim(mp_size: int) -> List[int]:
        mp_dims = list()

        dim = -1
        current_mp_size = 1

        while current_mp_size < mp_size:
            dim += 1
            current_mp_size *= Model.network.npus_count[dim]
            mp_dims.append(dim)

        return mp_dims

    @staticmethod
    def _get_mp_npus_count(mp_size: int,
                           mp_dim: List[int]) -> List[int]:
        if len(mp_dim) <= 0:
            return list()

        mp_npus_count = Model.network.npus_count[:len(mp_dim) - 1]

        last_mp_size = mp_size // math.prod(mp_npus_count)
        mp_npus_count.append(last_mp_size)

        return mp_npus_count

    @staticmethod
    def _get_dp_dim(dp_size: int) -> List[int]:
        dp_dims = list()

        dim = Model.network.dims_count
        current_dp_size = 1

        while current_dp_size < dp_size:
            dim -= 1
            current_dp_size *= Model.network.npus_count[dim]
            dp_dims.insert(0, dim)

        return dp_dims

    @staticmethod
    def _get_dp_npus_count(dp_size: int,
                           dp_dim: List[int]) -> List[int]:
        if len(dp_dim) == 1:
            dp_npus_count = list()
        else:
            dp_npus_count = Model.network.npus_count[-len(dp_dim) + 1:]
        last_dp_size = dp_size // math.prod(dp_npus_count)
        dp_npus_count.insert(0, last_dp_size)
        return dp_npus_count

    @staticmethod
    def _collective(collective_type: Collective,
                    processing_dims: List[int],
                    npus_count: List[int],
                    collective_size: float) -> torch.float64:
        if collective_type == Collective.NoComm:
            return 0.0

        if len(processing_dims) <= 0:
            return 0.0

        if collective_type == Collective.ReduceScatter:
            messages = list()  # todo: implement message size logic
            assert False, "Not implemented yet"
            exit(-1)

        if collective_type == Collective.AllGather:
            dims_count = len(processing_dims)

            message_size = [collective_size]
            for i in range(dims_count - 1, 0, -1):
                npus = npus_count[i]
                new_message_size = message_size[0] * npus
                message_size = [new_message_size] + message_size

            messages = list()
            for i in range(dims_count):
                npus = npus_count[i]

                new_message_size = message_size[i] * (npus - 1)
                messages.append(new_message_size)

        if collective_type == Collective.AllReduce:
            dims_count = len(processing_dims)

            # calculate message size, per NPU
            message_size = [collective_size]
            for i in range(1, dims_count):
                npus = npus_count[i - 1]
                new_message_size = message_size[-1] / npus
                message_size.append(new_message_size)

            # calculate message size, at network level
            messages = list()
            for i in range(dims_count):
                npus = npus_count[i]

                new_message_size = (message_size[i] / npus) * (npus - 1) * 2
                messages.append(new_message_size)

        if collective_type == Collective.AllToAll:
            messages = list()  # todo: implement message size logic
            assert False, "Not implemented yet"
            exit(-1)

        # get processing time of each dim
        times = list()
        for i in range(dims_count):
            dim = processing_dims[i]
            time = messages[i] / Model.bandwidths[dim]
            times.append(time)

        return max(times)
