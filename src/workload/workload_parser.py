import os
from src.workload.layer import Layer
from src.workload.layer_entry import LayerEntry
from src.helper.typing import *
from src.workload.workload import Workload


class WorkloadParser:
    def __init__(self, path: str):
        self.path = path

    def parse(self) -> Workload:
        assert os.path.exists(self.path), "Workload file doesn't exist"

        with open(self.path, mode='r') as fp:
            header_str = fp.readline()
            layers_count_str = fp.readline()
            layers_str = fp.readlines()

        header = self._parse_header(header_str=header_str)
        layers_count = self._parse_layers_count(layers_count_str=layers_count_str)
        layers = self._parse_layers(layers_str=layers_str)

        return Workload(header=header,
                        layers_count=layers_count,
                        layers=layers)

    @staticmethod
    def _parse_header(header_str: str):
        return header_str.strip()

    @staticmethod
    def _parse_layers_count(layers_count_str: str) -> int:
        layers_count = layers_count_str.strip()
        layers_count = int(layers_count)
        return layers_count


    @staticmethod
    def _parse_layers(layers_str: List[str]) -> List[Layer]:
        layers = list()

        for layer_str in layers_str:
            layer = WorkloadParser._parse_layer(layer_str=layer_str)
            layers.append(layer)

        return layers

    @staticmethod
    def _parse_layer(layer_str: str) -> Layer:
        layer = layer_str.split()

        layer_name = layer[0]
        forward = WorkloadParser._create_layer_entry(info=layer[2:5])
        input_grad = WorkloadParser._create_layer_entry(info=layer[5:8])
        weight_grad = WorkloadParser._create_layer_entry(info=layer[8:11])

        return Layer(name=layer_name,
                     forward=forward,
                     input_grad=input_grad,
                     weight_grad=weight_grad)

    @staticmethod
    def _parse_collective(collective_str: str) -> Collective:
        if collective_str == 'NONE':
            return Collective.NoComm

        if collective_str == 'REDUCESCATTER':
            return Collective.ReduceScatter

        if collective_str == 'ALLGATHER':
            return Collective.AllGather

        if collective_str == 'ALLREDUCE':
            return Collective.AllReduce

        if collective_str == 'ALLTOALL':
            return Collective.AllToAll

        assert False, "Shouldn't reach here"
        exit(-1)

    @staticmethod
    def _create_layer_entry(info: List[str]) -> LayerEntry:
        comp_time = float(info[0])
        comm_type = WorkloadParser._parse_collective(collective_str=info[1])
        comm_size = float(info[2])

        return LayerEntry(compute_time=comp_time,
                          comm_type=comm_type,
                          comm_size=comm_size)
