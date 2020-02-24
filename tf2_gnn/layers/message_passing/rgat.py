"""Relation graph attention network layer."""
from typing import Dict, List, Tuple, Any

import tensorflow as tf
from dpu_utils.tf2utils import unsorted_segment_log_softmax

from .message_passing import MessagePassing, MessagePassingInput, register_message_passing_implementation


@register_message_passing_implementation
class RGAT(MessagePassing):
    """Compute new graph states by neural message passing using attention. This generalises
    the original GAT model (Velickovic et al., https://arxiv.org/pdf/1710.10903.pdf)
    to multiple edge types by using different weights for different edge types.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    In the setting for a single attention head, we compute new states as follows:
        h^t_{v, \ell} := W_\ell h^t_v
        e_{u, \ell, v} := LeakyReLU(\alpha_\ell^T * concat(h^t_{u, \ell}, h^t_{v, \ell}))
        a_v := softmax_{\ell, u with (u, v) \in A_\ell}(e_{u, \ell, v})
        h^{t+1}_v := \sigma(\sum_{ell, (u, v) \in A_\ell}
                                a_v_{u, \ell} * h^_{u, \ell})
    The learnable parameters of this are the W_\ell \in R^{D, D} and \alpha_\ell \in R^{2*D}.

    In practice, we use K attention heads, computing separate, partial new states h^{t+1}_{v,k}
    and compute h^{t+1}_v as the concatentation of the partial states.
    For this, we reduce the shape of W_\ell to R^{D, D/K} and \alpha_\ell to R^{2*D/K}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * K: number of attention heads
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
    >>> params = RGAT.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = RGAT(params)
    >>> output = layer(MessagePassingInput(node_embeddings, adjacency_lists))
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        these_hypers = {
            "num_heads": 3,
        }
        mp_hypers = super().get_default_hyperparameters()
        mp_hypers.update(these_hypers)
        return mp_hypers

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self._num_heads: int = params["num_heads"]
        self._edge_type_to_message_computation_layer: List[tf.keras.layers.Layer] = []
        self._edge_type_to_attention_parameters: List[tf.Variable] = []

    def build(self, input_shapes: MessagePassingInput):
        node_embedding_shapes = input_shapes.node_embeddings
        adjacency_list_shapes = input_shapes.adjacency_lists
        num_edge_types = len(adjacency_list_shapes)
        per_head_dim = self._hidden_dim // self._num_heads

        for i in range(num_edge_types):
            with tf.name_scope(f"edge_type_{i}"):
                mp_layer = tf.keras.layers.Dense(
                    self._hidden_dim, use_bias=False, name="Edge_weight_{}".format(i)
                )
                mp_layer.build(tf.TensorShape((None, node_embedding_shapes[-1])))
                self._edge_type_to_message_computation_layer.append(mp_layer)

                attention_weights = self.add_weight(
                    name="Edge_attention_parameters_{}".format(i),
                    shape=(self._num_heads, 2 * per_head_dim),
                    trainable=True,
                )
                self._edge_type_to_attention_parameters.append(attention_weights)

        super().build(input_shapes)

    def _message_function(
        self,
        edge_source_states: tf.Tensor,
        edge_target_states: tf.Tensor,
        num_incoming_to_node_per_message: tf.Tensor,
        edge_type_idx: int,
        training: bool,
    ) -> tf.Tensor:
        per_head_dim = self._hidden_dim // self._num_heads

        # Actually do the message calculation:
        per_head_transformed_source_states = tf.reshape(
            self._edge_type_to_message_computation_layer[edge_type_idx](edge_source_states),
            shape=(-1, self._num_heads, per_head_dim),
        )  # Shape [E, K, H/K]
        per_head_transformed_target_states = tf.reshape(
            self._edge_type_to_message_computation_layer[edge_type_idx](edge_target_states),
            shape=(-1, self._num_heads, per_head_dim),
        )  # Shape [E, K, H/K]

        per_head_transformed_states = tf.concat(
            [per_head_transformed_source_states, per_head_transformed_target_states], axis=-1
        )  # Shape [E, K, 2*H/K]

        per_head_attention_scores = tf.nn.leaky_relu(
            tf.einsum(
                "vki,ki->vk",
                per_head_transformed_states,
                self._edge_type_to_attention_parameters[edge_type_idx],
            )
        )  # Shape [E, K]

        return (per_head_transformed_source_states, per_head_attention_scores)

    def _compute_new_node_embeddings(
        self,
        cur_node_embeddings: tf.Tensor,
        messages_per_type: List[Tuple[tf.Tensor, tf.Tensor]],
        edge_type_to_message_targets: List[tf.Tensor],
        num_nodes: tf.Tensor,
        training: bool,
    ):
        per_head_messages_per_type, per_head_attention_scores_per_type = zip(*messages_per_type)

        per_head_messages = tf.concat(per_head_messages_per_type, axis=0)  # Shape [M, K, H/K]
        per_head_attention_scores = tf.concat(
            per_head_attention_scores_per_type, axis=0
        )  # Shape [M, K]
        message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

        head_to_aggregated_messages = []  # list of tensors of shape [V, H/K]
        for head_idx in range(self._num_heads):
            # Compute the softmax over all the attention scores for all messages going to this state:
            attention_scores = tf.concat(
                per_head_attention_scores[:, head_idx], axis=0
            )  # Shape [M]
            attention_values = tf.exp(
                unsorted_segment_log_softmax(
                    logits=attention_scores, segment_ids=message_targets, num_segments=num_nodes
                )
            )  # Shape [M]
            messages = per_head_messages[:, head_idx, :]  # Shape [M, H/K]
            # Compute weighted sum per target node for this head:
            head_to_aggregated_messages.append(
                tf.math.unsorted_segment_sum(
                    data=tf.expand_dims(attention_values, -1) * messages,
                    segment_ids=message_targets,
                    num_segments=num_nodes,
                )
            )

        aggregated_messages = tf.concat(head_to_aggregated_messages, axis=-1)  # Shape [V, H]
        return self._activation_fn(aggregated_messages)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
