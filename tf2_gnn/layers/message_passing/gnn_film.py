"""Graph neural network layer using feature-wise linear modulation to compute edge messages."""
from typing import Dict, List, Any

import tensorflow as tf
from dpu_utils.tf2utils import MLP

from .gnn_edge_mlp import GNN_Edge_MLP
from .message_passing import MessagePassing, MessagePassingInput,register_message_passing_implementation
from tf2_gnn.utils.constants import SMALL_NUMBER


@register_message_passing_implementation
class GNN_FiLM(GNN_Edge_MLP):
    """Compute new graph states by neural message passing modulated by the target state.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sum_\ell
                     \sum_{(u, v) \in A_\ell}
                        \sigma(1/c_{v,\ell} * \alpha_{\ell,v} * (W_\ell * h^t_u) + \beta_{\ell,v})
        \alpha_{\ell,v} := F_{\ell,\alpha} * h^t_v
        \beta_{\ell,v} := F_{\ell,\beta} * h^t_v
        c_{\v,\ell} is usually 1 (but could also be the number of incoming edges).
    The learnable parameters of this are the W_\ell, F_{\ell,\alpha}, F_{\ell,\beta} \in R^{D, D}.

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
    >>> params = GNN_FiLM.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = GNN_FiLM(params)
    >>> output = layer(MessagePassingInput(node_embeddings, adjacency_lists))
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        these_hypers = {
            "use_target_state_as_input": False,
            "normalize_by_num_incoming": False,
            "num_edge_MLP_hidden_layers": 0,
            "film_parameter_MLP_hidden_layers": [],
        }
        mp_hypers = super().get_default_hyperparameters()
        mp_hypers.update(these_hypers)
        return mp_hypers

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self._film_parameter_MLP_hidden_layers = params["film_parameter_MLP_hidden_layers"]

        self._edge_type_film_layer_computations: List[tf.keras.layers.Layer] = []

    def build(self, input_shapes: MessagePassingInput):
        node_embedding_shapes = input_shapes.node_embeddings
        adjacency_list_shapes = input_shapes.adjacency_lists
        num_edge_types = len(adjacency_list_shapes)

        for i in range(num_edge_types):
            with tf.name_scope(f"edge_type_{i}-FiLM"):
                film_mlp = MLP(
                    out_size=2 * self._hidden_dim,
                    hidden_layers=self._film_parameter_MLP_hidden_layers,
                )
                film_mlp.build(tf.TensorShape((None, node_embedding_shapes[-1])))
                self._edge_type_film_layer_computations.append(film_mlp)

        super().build(input_shapes)

    def _message_function(
        self,
        edge_source_states: tf.Tensor,
        edge_target_states: tf.Tensor,
        num_incoming_to_node_per_message: tf.Tensor,
        edge_type_idx: int,
        training: bool,
    ) -> tf.Tensor:
        messages = super()._message_function(
            edge_source_states,
            edge_target_states,
            num_incoming_to_node_per_message,
            edge_type_idx,
            training,
        )

        film_weights = self._edge_type_film_layer_computations[edge_type_idx](
            edge_target_states, training
        )
        per_message_film_gamma_weights = film_weights[:, : self._hidden_dim]  # Shape [E, D]
        per_message_film_beta_weights = film_weights[:, self._hidden_dim :]  # Shape [E, D]

        modulated_messages = (
            per_message_film_gamma_weights * messages + per_message_film_beta_weights
        )
        return modulated_messages


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
