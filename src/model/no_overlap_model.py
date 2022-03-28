import torch
from src.model.model import Model


class NoOverlapModel(Model):
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
            training_time += Model._collective(collective_type=layer.forward.comm_type,
                                               processing_dims=self.mp_dim,
                                               npus_count=self.mp_npus_count,
                                               collective_size=layer.forward.comm_size)

        # backprop
        for i in range(len(self.workload.layers) - 1, -1, -1):
            layer = self.workload.layers[i]

            # input grad
            training_time += layer.input_grad.compute_time
            training_time += Model._collective(collective_type=layer.input_grad.comm_type,
                                               processing_dims=self.mp_dim,
                                               npus_count=self.mp_npus_count,
                                               collective_size=layer.input_grad.comm_size)
            # weight grad
            training_time += layer.weight_grad.compute_time
            training_time += Model._collective(collective_type=layer.weight_grad.comm_type,
                                               processing_dims=self.dp_dim,
                                               npus_count=self.dp_npus_count,
                                               collective_size=layer.weight_grad.comm_size)

        return training_time
