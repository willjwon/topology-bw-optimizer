from src.model.model_no_overlap import ModelNoOverlap
from src.runner.model_optimizer import ModelOptimizer
from src.network.network_parser import NetworkParser
from src.model.model import Model


def main():
    # metadata
    network_path = '../../input/network/4d.json'

    t17b_path = '../../input/workload/t17b.txt'
    t1t_path = '../../input/workload/t1t.txt'
    gpt3_path = '../../input/workload/gpt3.txt'

    # parse network
    network_parser = NetworkParser(path=network_path)
    network = network_parser.parse()

    # create and set model
    Model.init(network=network)
    t17b_model = ModelNoOverlap(workload_path=t17b_path,
                                mp_size=1,
                                dp_size=1024)
    gpt3_model = ModelNoOverlap(workload_path=gpt3_path,
                                mp_size=16,
                                dp_size=64)
    t1t_model = ModelNoOverlap(workload_path=t1t_path,
                               mp_size=128,
                               dp_size=8)
    models = [t17b_model, gpt3_model, t1t_model]

    # Create and run optimizer
    optimizer = ModelOptimizer(models=models,
                               weights=None,
                               lr=5e-3,
                               l2_break=1e-6)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)

    # Print the result
    print("\nFound BW Assignment:\n\t", end="")
    models[0].print_bandwidth()
    print()


if __name__ == '__main__':
    main()
