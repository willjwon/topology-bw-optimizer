import sys
from src.cost.cost_calculator import CostCalculator
from src.model.model import Model
from src.model.model_no_overlap import ModelNoOverlap
from src.model.model_overlap import ModelOverlap
from src.model.model_cost_no_overlap import ModelCostNoOverlap
from src.model.model_cost_overlap import ModelCostOverlap
from src.network.network_parser import NetworkParser
from src.runner.model_optimizer import ModelOptimizer


def main():
    # metadata
    network_path = sys.argv[1]
    cost_model_path = sys.argv[2]
    bandwidth = float(sys.argv[3])
    bw_target = sys.argv[4]
    training_loop = sys.argv[5]
    workload_paths = [sys.argv[6:9]]
    mp_size = [1, 16, 128]
    dp_size = [1024, 64, 8]

    lr = 5e-3 if bw_target == 'Perf' else 3e-2

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()
    network.total_bandwidth = bandwidth

    # cost calculator
    cost_calculator = CostCalculator(cost_model_path=cost_model_path,
                                     network=network)

    # create and set model
    Model.init(network=network, cost_calculator=cost_calculator)
    models = list()
    for i in range(3):
        if training_loop == 'NoOverlap':
            if bw_target == 'Perf':
                model.append(ModelNoOverlap(workload_path=workload_paths[i],
                                        mp_size=mp_size[i],
                                        dp_size=dp_size[i]))
            else:
                model.append(ModelCostNoOverlap(workload_path=workload_paths[i],
                                        mp_size=mp_size[i],
                                        dp_size=dp_size[i]))
        else:
            if bw_target == 'Perf':
                model.append(ModelOverlap(workload_path=workload_paths[i],
                                        mp_size=mp_size[i],
                                        dp_size=dp_size[i]))
            else:
                model.append(ModelCostOverlap(workload_path=workload_paths[i],
                                        mp_size=mp_size[i],
                                        dp_size=dp_size[i]))

    # Create and run optimizer
    optimizer = ModelOptimizer(models=models,
                               weights=None,
                               lr=lr,
                               l2_break=None)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    models[0].print_bandwidth()
    print()


if __name__ == '__main__':
    main()
