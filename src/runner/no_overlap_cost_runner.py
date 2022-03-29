from src.model.model_cost_no_overlap import ModelCostNoOverlap
from src.runner.model_optimizer import ModelOptimizer
from src.network.network_parser import NetworkParser
from src.model.model import Model
from src.cost.cost_calculator import CostCalculator


def main():
    # metadata
    workload_path = '../../input/workload/t17b.txt'
    network_path = '../../input/network/4d.json'
    cost_model_path = '../../input/cost/cost.json'

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()

    # create cost calculator
    cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                     network=network)

    # create and set model
    Model.init(network=network, cost_calculator=cost_calculator)
    model = ModelCostNoOverlap(workload_path=workload_path,
                               mp_size=1,
                               dp_size=1024)

    # Create and run optimizer
    optimizer = ModelOptimizer(models=[model],
                               weights=None,
                               lr=3e-2,
                               l2_break=1e-10)
    optimizer.optimize(steps_count=500000,
                       print_step=10000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    model.print_bandwidth()
    print()


if __name__ == '__main__':
    main()
