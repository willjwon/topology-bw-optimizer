from enum import Enum, auto
from typing import List, Optional, Tuple


class Topology(Enum):
    Ring = auto()
    FullyConnected = auto()
    Switch = auto()


class NetworkType(Enum):
    Tile = auto()
    Package = auto()
    Node = auto()
    Pod = auto()


class CostElement(Enum):
    Link = auto()
    Nic = auto()
    Switch = auto()


class Collective(Enum):
    NoComm = auto()
    ReduceScatter = auto()
    AllGather = auto()
    AllReduce = auto()
    AllToAll = auto()
