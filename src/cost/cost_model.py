from src.helper.typing import *


class CostModel:
    def __init__(self):
        self.cost_dict = dict()

        for network in NetworkType:
            self.cost_dict[network] = dict()

            for cost in CostElement:
                self.cost_dict[network][cost] = None

    def set(self,
            network_type: NetworkType,
            cost_element: CostElement,
            cost: float) -> None:
        self.cost_dict[network_type][cost_element] = cost

    def cost(self,
             network_type: NetworkType,
             cost_element: CostElement) -> Optional[float]:
        price = self.cost_dict[network_type][cost_element]
        return price
