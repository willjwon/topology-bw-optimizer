import torch
from src.model.no_overlap_model import NoOverlapModel
from src.model.model import Model


class NoOverlapModelCost(NoOverlapModel):
    cost_calculator = None

    def __init__(self,
                 workload_path: str,
                 mp_size: int,
                 dp_size: int):
        super().__init__(workload_path=workload_path,
                         mp_size=mp_size,
                         dp_size=dp_size)

    def training_time(self) -> torch.tensor:
        # optimizer perf-per-cost
        time = super().training_time()
        cost = Model.cost_calculator.cost(bandwidths=Model.bandwidths)
        return time * cost
