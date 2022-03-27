from src.model.no_overlap_model import NoOverlapModel
from src.runner.model_optimizer import ModelOptimizer


def main():
    # metadata
    workload_path = '../../input/workload/t17b.txt'
    network_path = '../../input/network/network.json'

    model = NoOverlapModel(workload_path=workload_path,
                           network_path=network_path,
                           mp_dim=None,
                           dp_dim=[0, 1, 2, 3])
    optimizer = ModelOptimizer(model=model,
                               lr=5e-3)
    optimizer.optimize(steps_count=500000,
                       print_step=5000)


if __name__ == '__main__':
    main()
