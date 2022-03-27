from src.helper.typing import *


class Network:
    def __init__(self,
                 total_bandwidth: float,
                 npus_count: List[int],
                 network_type: List[NetworkType],
                 topology: List[Topology]):
        self.total_bandwidth = total_bandwidth
        self.npus_count = npus_count
        self.network_type = network_type
        self.topology = topology
