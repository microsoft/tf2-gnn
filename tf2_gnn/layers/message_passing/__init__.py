from .message_passing import MessagePassing, MessagePassingInput, MESSAGE_PASSING_IMPLEMENTATIONS
from .rgat import RGAT
from .rgcn import RGCN
from .rgin import RGIN
from .ggnn import GGNN
from .gnn_edge_mlp import GNN_Edge_MLP
from .gnn_film import GNN_FiLM

def get_message_passing_class(message_calculation_class_name: str):
    calculation_class = MESSAGE_PASSING_IMPLEMENTATIONS.get(message_calculation_class_name.lower())
    if calculation_class is None:
        raise ValueError(f"Unknown message passing type: {message_calculation_class_name}")
    return calculation_class

def get_known_message_passing_classes():
    for message_passing_implementation in MESSAGE_PASSING_IMPLEMENTATIONS.values():
        yield message_passing_implementation.__name__
