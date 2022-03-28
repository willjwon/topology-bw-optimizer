import torch
import math
from src.network.network import Network
from src.cost.cost_parser import CostParser
from src.helper.typing import *


class CostCalculator:
    def __init__(self,
                 cost_model_path: str,
                 network: Network):
        # cost model
        cost_parser = CostParser(path=cost_model_path)
        self.cost_model = cost_parser.parse()

        # network
        self.network = network
        self.total_npus_count = self._get_total_npus_count()

    def cost(self, bandwidths: torch.tensor) -> torch.tensor:
        total_cost = torch.tensor(0.0, dtype=torch.float64)

        child_npus_count = 1

        for dim in range(self.network.dims_count):
            # get topologies count
            npus_count = self.network.npus_count[dim]
            topology = self.network.topology[dim]
            topologies_count = self._get_topologies_count(topology=topology,
                                                          npus_count=npus_count,
                                                          child_npus_count=child_npus_count)

            # get topology cost
            network_type = self.network.network_type[dim]
            bandwidth = bandwidths[dim]
            topology_cost = self._get_topology_cost(topology=topology,
                                                    network_type=network_type,
                                                    npus_count=npus_count,
                                                    child_npus_count=child_npus_count,
                                                    bandwidth=bandwidth)
            # update total cost
            dim_cost = topologies_count * topology_cost
            total_cost += dim_cost

            # update child NPUs
            child_npus_count *= npus_count

        return total_cost

    def _get_total_npus_count(self) -> int:
        return math.prod(self.network.npus_count)

    def _get_topologies_count(self,
                              topology: Topology,
                              npus_count: int,
                              child_npus_count: int) -> int:
        if topology == Topology.Switch:
            return self.total_npus_count // child_npus_count

        if topology == Topology.Ring or topology == Topology.FullyConnected:
            return self.total_npus_count // npus_count

        assert False, "Should not reach here"
        exit(-1)

    @staticmethod
    def _get_link_bandwidth(topology: Topology,
                            npus_count: int,
                            bandwidth: float) -> float:
        if topology == Topology.Ring:
            return bandwidth / 2

        if topology == Topology.FullyConnected:
            return bandwidth / (npus_count - 1)

        if topology == Topology.Switch:
            return bandwidth

        assert False, "Should not reach here"
        exit(-1)

    def _get_topology_cost(self,
                           topology: Topology,
                           network_type: NetworkType,
                           npus_count: int,
                           child_npus_count: int,
                           bandwidth: float) -> float:
        total_cost = 0

        # calculate link cost
        link_bandwidth = self._get_link_bandwidth(topology=topology,
                                                  npus_count=npus_count,
                                                  bandwidth=bandwidth)
        links_count = self._get_links_count(topology=topology,
                                            npus_count=npus_count,
                                            child_npus_count=child_npus_count)
        link_cost = self.cost_model.cost(network_type=network_type,
                                         cost_element=CostElement.Link)
        total_cost += (links_count * link_bandwidth * link_cost)

        # if switch, add switch cost
        if topology == Topology.Switch:
            switch_cost = self.cost_model.cost(network_type=network_type,
                                               cost_element=CostElement.Switch)
            total_cost += (links_count * link_bandwidth * switch_cost)

            # also, if nic exists, add nic cost
            nic_cost = self.cost_model.cost(network_type=network_type,
                                            cost_element=CostElement.Nic)
            if nic_cost is not None:
                total_cost += (links_count * link_bandwidth * nic_cost)

        return total_cost

    @staticmethod
    def _get_links_count(topology: Topology,
                         npus_count: int,
                         child_npus_count: int) -> int:
        if topology == Topology.Ring:
            return npus_count

        if topology == Topology.FullyConnected:
            return (npus_count * (npus_count - 1)) // 2

        if topology == Topology.Switch:
            return child_npus_count

        assert False, "Should not reach here."
        exit(-1)
