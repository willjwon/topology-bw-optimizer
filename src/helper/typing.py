from enum import Enum, auto
from typing import Dict


class Topology(Enum):
    Ring = auto()
    FullyConnected = auto()
    Switch = auto()


class NetworkType(Enum):
    Tile = auto()
    Package = auto()
    Node = auto()
    Pod = auto()
