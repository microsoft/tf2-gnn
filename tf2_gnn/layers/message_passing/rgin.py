"""Relation Graph Isomorphism Network message propogation layer."""
from typing import Dict, List, Any, Optional

import tensorflow as tf
from dpu_utils.tf2utils import MLP

from .message_passing import MessagePassing, MessagePassingInput, register_message_passing_implementation
from .gnn_edge_mlp import GNN_Edge_MLP
from tf2_gnn.utils.constants import SMALL_NUMBER


@register_message_passing_implementation
class RGIN(GNN_Edge_MLP):
    """Compute new graph states by neural message passing using MLPs for state updates
    and message computation.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sigma(MLP_{aggr}(\sum_\ell \sum_{(u, v) \in A_\ell} MLP_\ell(h^t_u)))
    The learnable parameters of this are the MLPs MLP_\ell.
    This is derived from Cor. 6 of arXiv:1810.00826, instantiating the functions f, \phi
    with _separate_ MLPs. This is more powerful than the GIN formulation in Eq. (4.1) of
    arXiv:1810.00826, as we want to be able to distinguish graphs of the form
     G_1 = (V={1, 2, 3}, E_1={(1, 2)}, E_2={(3, 2)})
    and
     G_2 = (V={1, 2, 3}, E_1={(3, 2)}, E_2={(1, 2)})
    from each other. If we would treat all edges the same,
    G_1.E_1 \cup G_1.E_2 == G_2.E_1 \cup G_2.E_2 would imply that the two graphs
    become indistuingishable.
    Hence, we introduce per-edge-type MLPs, which also means that we have to drop
    the optimisation of modelling f \circ \phi by a single MLP used in the original
    GIN formulation.

    Note that RGIN is implemented as a special-case of GNN_Edge_MLP, setting some
    different default hyperparameters and adding a different message aggregation
    function, but re-using the message passing functionality.

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
    >>> params = RGIN.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = RGIN(params)
    >>> output = layer(MessagePassingInput(node_embeddings, adjacency_lists))
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        these_hypers = {
            "use_target_state_as_input": False,
            "num_edge_MLP_hidden_layers": 1,
            "num_aggr_MLP_hidden_layers": None,
        }
        gnn_edge_mlp_hypers = super().get_default_hyperparameters()
        gnn_edge_mlp_hypers.update(these_hypers)
        return gnn_edge_mlp_hypers

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self._num_aggr_MLP_hidden_layers: Optional[int] = params["num_aggr_MLP_hidden_layers"]
        self._aggregation_mlp: Optional[MLP] = None

    def build(self, input_shapes: MessagePassingInput):
        node_embedding_shapes = input_shapes.node_embeddings
        if self._num_aggr_MLP_hidden_layers is not None:
            with tf.name_scope("aggregation_MLP"):
                self._aggregation_mlp = MLP(
                    out_size=self._hidden_dim,
                    hidden_layers=[self._hidden_dim] * self._num_aggr_MLP_hidden_layers,
                )
                self._aggregation_mlp.build(tf.TensorShape((None, self._hidden_dim)))
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
        if self._aggregation_mlp is not None:
            aggregated_messages = self._aggregation_mlp(aggregated_messages, training)

        return self._activation_fn(aggregated_messages)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
