from src.cost.cost_calculator import CostCalculator
from src.model.model import Model
from src.model.model_overlap import ModelOverlap
from src.network.network_parser import NetworkParser
from src.runner.model_optimizer import ModelOptimizer


def main():
    # metadata
    network_path = '../../input/network/4d.json'
    cost_model_path = '../../input/cost/cost.json'

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()

    # cost calculator
    cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                     network=network)

    # create and set model
    Model.init(network=network, cost_calculator=cost_calculator)
    models = list()
    models.append(
        ModelOverlap(workload_path='../../input/workload_not_fused/transformer_1T.txt',
                     mp_size=128,
                     dp_size=8)
    )

    # Create and run optimizer
    optimizer = ModelOptimizer(models=models,
                               weights=None,
                               lr=5e-3,
                               l2_break=None)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    models[0].print_bandwidth()
    print()


if __name__ == '__main__':
    main()
