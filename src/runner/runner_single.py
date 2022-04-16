import sys
from src.cost.cost_calculator import CostCalculator
from src.model.model import Model
from src.model.model_no_overlap_fwd_in_bckwd import ModelNoOverlapFwdInBckwd
from src.model.model_overlap_fwd_in_bckwd import ModelOverlapFwdInBckwd
from src.model.model_cost_no_overlap_fwd_in_bckwd import ModelCostNoOverlapFwdInBwd
from src.model.model_cost_overlap_fwd_in_bckwd import ModelCostOverlapFwdInBckwd
from src.network.network_parser import NetworkParser
from src.runner.model_optimizer import ModelOptimizer


def main():
    # metadata
    network_path = sys.argv[1]
    cost_model_path = sys.argv[2]
    workload_path = sys.argv[3]
    bandwidth = float(sys.argv[4])
    bw_target = sys.argv[5]
    training_loop = sys.argv[6]
    mp_size = int(sys.argv[7])
    dp_size = int(sys.argv[8])

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
    model = None
    if training_loop == 'NoOverlap':
        if bw_target == 'Perf':
            model = ModelNoOverlapFwdInBckwd(workload_path=workload_path,
                                    mp_size=mp_size,
                                    dp_size=dp_size)
        else:
            model = ModelCostNoOverlapFwdInBwd(workload_path=workload_path,
                                    mp_size=mp_size,
                                    dp_size=dp_size)
    else:
        if bw_target == 'Perf':
            model = ModelOverlapFwdInBckwd(workload_path=workload_path,
                                    mp_size=mp_size,
                                    dp_size=dp_size)
        else:
            model = ModelCostOverlapFwdInBckwd(workload_path=workload_path,
                                    mp_size=mp_size,
                                    dp_size=dp_size)

    # Create and run optimizer
    optimizer = ModelOptimizer(models=[model],
                               weights=None,
                               lr=lr,
                               l2_break=None)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    model.print_bandwidth()
    print()


if __name__ == '__main__':
    main()
