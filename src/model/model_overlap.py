import torch
from src.model.model import Model


class ModelOverlap(Model):
    def training_time(self) -> torch.tensor:
        # calculate time
        training_time = torch.tensor(0, dtype=torch.float64)

        # forward pass
        for layer in self.workload.layers:
            training_time += layer.forward.compute_time
            training_time += Model._collective(collective_type=layer.forward.comm_type,
                                               processing_dims=self.mp_dim,
                                               npus_count=self.mp_npus_count,
                                               collective_size=layer.forward.comm_size)

        # backprop
        for i in range(len(self.workload.layers) - 1, -1, -1):
            layer = self.workload.layers[i]

            # input grad
            ig_compute = layer.input_grad.compute_time
            ig_comm = Model._collective(collective_type=layer.input_grad.comm_type,
                                        processing_dims=self.mp_dim,
                                        npus_count=self.mp_npus_count,
                                        collective_size=layer.input_grad.comm_size)
            ig_time = ig_compute + ig_comm

            # weight grad
            wg_compute = layer.weight_grad.compute_time
            wg_comm = Model._collective(collective_type=layer.weight_grad.comm_type,
                                        processing_dims=self.dp_dim,
                                        npus_count=self.dp_npus_count,
                                        collective_size=layer.weight_grad.comm_size)
            wg_time = ig_compute + (wg_compute + wg_comm)  # wg launches after ig compute

            layer_time = max(ig_time, wg_time)
            training_time += layer_time

        return training_time
