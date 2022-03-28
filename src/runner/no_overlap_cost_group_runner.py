from src.model.no_overlap_model_cost import NoOverlapModelCost
from src.runner.model_optimizer import ModelOptimizer
from src.network.network_parser import NetworkParser
from src.model.model import Model
from src.cost.cost_calculator import CostCalculator


def main():
    print('GroupPerfPerCostOpt')

    # metadata
    network_path = '../../input/network/4d.json'
    cost_model_path = '../../input/cost/cost.json'

    t17b_path = '../../input/workload/t17b.txt'
    t1t_path = '../../input/workload/t1t.txt'
    gpt3_path = '../../input/workload/gpt3.txt'

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()

    # cost calculator
    cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                     network=network)

    # create and set model
    Model.init(network=network, cost_calculator=cost_calculator)
    t17b_model = NoOverlapModelCost(workload_path=t17b_path,
                                    mp_size=1,
                                    dp_size=1024)
    gpt3_model = NoOverlapModelCost(workload_path=gpt3_path,
                                    mp_size=16,
                                    dp_size=64)
    t1t_model = NoOverlapModelCost(workload_path=t1t_path,
                                   mp_size=128,
                                   dp_size=8)
    models = [t17b_model, gpt3_model, t1t_model]

    # Create and run optimizer
    optimizer = ModelOptimizer(models=models,
                               weights=None,
                               lr=3e-2,
                               l2_break=1e-8)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    models[0].print_bandwidth()
    print()


if __name__ == '__main__':
    main()
