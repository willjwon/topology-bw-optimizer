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

        return Network(total_bandwidth=total_bandwidth,
                       npus_count=npus_count,
                       network_type=network_type,
                       topology=topology)

    @staticmethod
    def _parse_network_type(network_type: List[str]) -> List[NetworkType]:
        result = list()

        for t in network_type:
            if t == 'Tile':
                result.append(NetworkType.Tile)
            elif t == 'Package':
                result.append(NetworkType.Package)
            elif t == 'Node':
                result.append(NetworkType.Node)
            elif t == 'Pod':
                result.append(NetworkType.Pod)
            else:
                assert False, "Network type not defined"
                exit(-1)

        return result

    @staticmethod
    def _parse_topology(topology: List[str]) -> List[Topology]:
        result = list()

        for t in topology:
            if t == 'Ring':
                result.append(Topology.Ring)
            elif t == 'FullyConnected':
                result.append(Topology.FullyConnected)
            elif t == 'Switch':
                result.append(Topology.Switch)
            else:
                assert False, "Topology not defined"
                exit(-1)

        return result
