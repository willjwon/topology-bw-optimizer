from abc import *
import torch
from src.workload.workload_parser import WorkloadParser
from src.network.network_parser import NetworkParser
from src.cost.cost_calculator import CostCalculator
from src.helper.typing import *
from src.model.model import Model
from src.cost.cost_calculator import CostCalculator


class ModelCost(Model):
    def __init__(self,
                 workload_path: str,
                 network_path: str,
                 cost_model_path: str,
                 mp_dim: Optional[List[int]],
                 dp_dim: Optional[List[int]]):
        super().__init__(workload_path, network_path, mp_dim, dp_dim)

        # cost calculator
        self.cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                              network=self.network)

    @abstractmethod
    def training_time(self) -> torch.tensor:
        self._sync_bandwidth()
        return
