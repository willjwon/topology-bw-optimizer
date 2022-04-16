import torch.optim.optimizer
from src.helper.typing import *
from src.model.model import Model


class ModelOptimizer:
    def __init__(self,
                 models: List[Model],
                 weights: Optional[List[float]],
                 lr: float,
                 l2_break: Optional[float] = None):
        self.models = models
        self.weights = self._normalize_weights(models_count=len(models),
                                               weights=weights)

        self.optimizer = torch.optim.Adam([self.models[0].torch_bandwidths],
                                          lr=lr)
        self.l2_break = l2_break

    def optimize(self,
                 steps_count: int,
                 print_step: Optional[int]) -> None:
        for step in range(1, steps_count + 1):
            old = self.models[0].bandwidths.detach().numpy().tolist()
            training_time = self._run_step()

            if print_step is not None:
                if step % print_step == 0:
                    progress = (step / steps_count) * 100
                    print(f"Step {step} / {steps_count} ({progress:.2f}%):")

                    print(f"\tEstimated Step Time: {training_time:.2f}")
                    print("\tCurrent BW: ", end="")
                    self.models[0].print_bandwidth()
                    print()

                    new = self.models[0].bandwidths.detach().numpy().tolist()
                    if self.l2_break is not None \
                            and self._l2_diff(old=old, new=new) < self.l2_break:
                        print(f"Optimization finished at step {step}")
                        break

    @staticmethod
    def _normalize_weights(models_count: int,
                           weights: Optional[List[int]]) -> List[float]:
        if weights is None:
            weights = [1] * models_count

        norm_weights = [x / sum(weights) for x in weights]
        return norm_weights

    def _run_step(self) -> torch.tensor:
        self.optimizer.zero_grad()

        self.models[0].sync_bandwidth()

        training_time = torch.tensor(0, dtype=torch.float64, requires_grad=False)
        for i in range(len(self.weights)):
            weight = self.weights[i]
            time = self.models[i].training_time()
            training_time += weight * time

        training_time.backward()
        self.optimizer.step()

        return training_time

    @staticmethod
    def _l2_diff(old: List[float], new: List[float]) -> float:
        dist = 0
        for i in range(len(old)):
            dist += (old[i] - new[i]) ** 2
        return dist
