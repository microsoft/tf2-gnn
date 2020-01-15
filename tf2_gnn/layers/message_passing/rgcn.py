"""Graph convolution layers."""
from typing import Dict, List, Any

import tensorflow as tf

from .gnn_edge_mlp import GNN_Edge_MLP
from .message_passing import MessagePassing, MessagePassingInput, register_message_passing_implementation
from tf2_gnn.utils.constants import SMALL_NUMBER


@register_message_passing_implementation
class RGCN(GNN_Edge_MLP):
    """Compute new graph states by neural message passing.
    This implements the R-GCN model:
    (Schlichtkrull et al., https://arxiv.org/pdf/1703.06103.pdf)
    for the case of few relations / edge types, i.e., we do not use the
    dimensionality-reduction tricks from section 2.2 of that paper.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sigma(\sum_\ell
                            \sum_{(u, v) \in A_\ell}
                               1/c_{v,\ell} * (W_\ell * h^t_u))
    c_{\v,\ell} is usually the number of \ell edges going into v.
    The learnable parameters of this are the W_\ell \in R^{D,D}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * L: number of different edge types
    * E: number of edges of a given edge type
    * D: input node representation dimension
    * H: output node representation dimension (set as hidden_dim)

    >>> node_embeddings = tf.random.normal(shape=(5, 3))
    >>> adjacency_lists = (
    ...    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[2, 3], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[3, 1]], dtype=tf.int32),
    ... )
    ...
    >>> params = RGCN.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = RGCN(params)
    >>> output = layer(MessagePassingInput(node_embeddings, adjacency_lists))
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        these_hypers = {
            "use_target_state_as_input": False,
            "normalize_by_num_incoming": True,
            "num_edge_MLP_hidden_layers": 0,
        }
        mp_hypers = super().get_default_hyperparameters()
        mp_hypers.update(these_hypers)
        return mp_hypers

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
