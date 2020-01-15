"""Gated graph neural network layer."""
from typing import Dict, List, Any

import tensorflow as tf

from .message_passing import MessagePassing, MessagePassingInput, register_message_passing_implementation
from .gnn_edge_mlp import GNN_Edge_MLP
from tf2_gnn.utils.constants import SMALL_NUMBER


@register_message_passing_implementation
class GGNN(GNN_Edge_MLP):
    """Compute new graph states by neural message passing and gated units on the nodes.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := Cell(h^t_v, \sum_\ell
                                 \sum_{(u, v) \in A_\ell}
                                     W_\ell * h^t_u)
    The learnable parameters of this are the recurrent Cell and the W_\ell \in R^{D,D}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * L: number of different edge types
    * E: number of edges of a given edge type
    * D: input node representation dimension
    * H: output node representation dimension (set as hidden_dim)

    NOTE: in this layer, the dimension of the node embedding must be equal to the hidden dimension!

    >>> node_embeddings = tf.random.normal(shape=(5, 12))
    >>> adjacency_lists = (
    ...    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[2, 3], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[3, 1]], dtype=tf.int32),
    ... )
    ...
    >>> params = GGNN.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = GGNN(params)
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
        self._recurrent_unit: tf.keras.layers.GRUCell = None

    def build(self, input_shapes: MessagePassingInput):
        node_embedding_shapes = input_shapes.node_embeddings
        self._recurrent_unit = tf.keras.layers.GRUCell(units=self._hidden_dim)
        self._recurrent_unit.build(tf.TensorShape((None, node_embedding_shapes[-1])))
        super().build(input_shapes)

    def _compute_new_node_embeddings(
        self,
        cur_node_embeddings: tf.Tensor,
        messages_per_type: List[tf.Tensor],
        edge_type_to_message_targets: List[tf.Tensor],
        num_nodes: tf.Tensor,
        training: bool,
    ):
        # Let M be the number of messages (sum of all E):
        message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]
        messages = tf.concat(messages_per_type, axis=0)  # Shape [M, H]

        aggregated_messages = self._aggregation_fn(
            data=messages, segment_ids=message_targets, num_segments=num_nodes
        )

        new_node_embeddings, _ = self._recurrent_unit(
            inputs=aggregated_messages,
            states=[cur_node_embeddings],
            training=training)

        return new_node_embeddings

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
