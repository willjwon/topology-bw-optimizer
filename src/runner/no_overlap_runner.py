from src.model.model_no_overlap import ModelNoOverlap
from src.runner.model_optimizer import ModelOptimizer
from src.network.network_parser import NetworkParser
from src.model.model import Model


def main():
    # metadata
    workload_path = '../../input/workload/t1t.txt'
    network_path = '../../input/network/4d.json'

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()

    # create and set model
    Model.init(network=network)
    model = ModelNoOverlap(workload_path=workload_path,
                           mp_size=128,
                           dp_size=8)

    # Create and run optimizer
    optimizer = ModelOptimizer(models=[model],
                               weights=None,
                               lr=5e-3,
                               l2_break=1e-6)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    model.print_bandwidth()
    print()


if __name__ == '__main__':
    main()
