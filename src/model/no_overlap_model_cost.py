import torch
from src.helper.typing import *
from src.cost.cost_calculator import CostCalculator
from src.model.no_overlap_model import NoOverlapModel


class NoOverlapModelCost(NoOverlapModel):
    def __init__(self,
                 workload_path: str,
                 network_path: str,
                 cost_model_path: str,
                 mp_dim: Optional[List[int]],
                 dp_dim: Optional[List[int]]):
        super().__init__(workload_path=workload_path,
                         network_path=network_path,
                         mp_dim=mp_dim,
                         dp_dim=dp_dim)

        # cost calculator
        self.cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                              network=self.network)

    def training_time(self) -> torch.tensor:
        # optimizer perf-per-cost
        time = super().training_time()
        cost = self.cost_calculator.cost(bandwidths=self.bandwidths)
        return time * cost
