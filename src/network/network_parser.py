from src.helper.typing import *
from src.network.network import Network
import json
import os


class NetworkParser:
    def __init__(self, path: str):
        self.path = path

    def parse(self) -> Network:
        assert os.path.exists(self.path), "File doesn't exist"

        with open(self.path, mode='r') as file:
            network_file = json.load(fp=file)

        total_bandwidth = network_file['total-bandwidth']
        npus_count = network_file['npus-count']
        network_type = self._parse_network_type(network_type=network_file['network-type'])
        topology = self._parse_topology(topology=network_file['topology'])
        dims_count = len(npus_count)

        return Network(dims_count=dims_count,
                       total_bandwidth=total_bandwidth,
                       npus_count=npus_count,
                       network_type=network_type,
                       topology=topology)

    @staticmethod
    def _parse_network_type(network_type: List[str]) -> List[NetworkType]:
        result = list()

        for t in network_type:
            result.append(NetworkType[t])

        return result

    @staticmethod
    def _parse_topology(topology: List[str]) -> List[Topology]:
        result = list()

        for t in topology:
            result.append(Topology[t])

        return result
