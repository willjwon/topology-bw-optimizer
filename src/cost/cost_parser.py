import os
import json
from src.cost.cost_model import CostModel
from src.helper.typing import *


class CostParser:
    def __init__(self, path: str):
        self.path = path

    def parse(self) -> CostModel:
        assert os.path.exists(self.path), "Cost file doesn't exist"

        with open(self.path, mode='r') as fp:
            cost_file = json.load(fp=fp)

        cost_model = CostModel()

        for network in cost_file:
            for element, cost in cost_file[network].items():
                network_type = NetworkType[network]
                cost_element = CostElement[element]

                cost_model.set(network_type=network_type,
                               cost_element=cost_element,
                               cost=cost)

        return cost_model
