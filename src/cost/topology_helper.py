from src.helper.typing import *


class TopologyHelper:
    def __init__(self):
        pass

    @staticmethod
    def total_links_count(topology: Topology,
                          npus_count: int) -> int:
        if topology == Topology.Ring:
            return npus_count

        elif topology == Topology.FullyConnected:

            links_count = npus_count * (npus_count - 1) // 2
            return links_count

        elif topology == Topology.Switch:
            return npus_count

        else:
            assert False, "Shouldn't reach here"
            exit(-1)
