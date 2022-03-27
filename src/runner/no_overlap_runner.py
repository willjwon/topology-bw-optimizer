from src.optimize.no_overlap_model import NoOverlapModel
from src.optimize.model_optimizer import ModelOptimizer


def main():
    # metadata
    workload_path = '../../input/workload/gpt3.txt'
    network_path = '../../input/network/network.json'

    model = NoOverlapModel(workload_path=workload_path,
                           network_path=network_path)
    optimizer = ModelOptimizer(model=model,
                               lr=1e-2)
    optimizer.optimize(steps_count=10000,
                       print_step=100)
    # print(model.bandwidths)


if __name__ == '__main__':
    main()
