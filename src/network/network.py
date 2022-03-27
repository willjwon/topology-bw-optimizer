from src.helper.typing import *


class Network:
    def __init__(self,
                 dims_count: int,
                 total_bandwidth: float,
                 npus_count: List[int],
                 network_type: List[NetworkType],
                 topology: List[Topology]):
        self.dims_count = dims_count
        self.total_bandwidth = total_bandwidth
        self.npus_count = npus_count
        self.network_type = network_type
        self.topology = topology

    def print(self):
        print("=======================")

        print(f"Total BW: {self.total_bandwidth}")
        print(f"{self.dims_count}-dim Network:")
        for i in range(self.dims_count):
            dim_index = i + 1
            npus_count = self.npus_count[i]
            network_type = self.network_type[i].name
            topology = self.topology[i].name
            print(f"\t{dim_index}D ({network_type}):\t{topology}({npus_count})")

        print("=======================")
