import torch.optim.optimizer
from src.helper.typing import *
from src.optimize.model import Model


class ModelOptimizer:
    def __init__(self,
                 model: Model,
                 lr: float):
        self.model = model
        self.optimizer = torch.optim.Adam([self.model.torch_bandwidths],
                                          lr=lr)

    def optimize(self,
                 steps_count: int,
                 print_step: Optional[int]) -> None:
        for step in range(1, steps_count + 1):
            training_time = self._run_step()

            if print_step is not None:
                if step % print_step == 0:
                    print(f"Step {step} / {steps_count}:")

                    print(f"\tEstimated Step Time: {training_time:.2f}")
                    print("\tCurrent BW: [ ", end="")
                    self.model.print_bandwidth()
                    print("]\n")

    def _run_step(self) -> torch.tensor:
        self.optimizer.zero_grad()
        training_time = self.model.training_time()
        training_time.backward()
        self.optimizer.step()
        return training_time
