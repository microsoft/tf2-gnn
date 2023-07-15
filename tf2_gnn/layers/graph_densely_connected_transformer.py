import tensorflow as tf

from tf2_gnn.utils.param_helpers import get_activation_function
from dpu_utils.tf2utils import unsorted_segment_softmax


# This helper function is currently not in use, as it is extremely slow
# in the computation graph. Instead, we pre-compute the list of index pairs.
def _all_shifted_pairs(lengths: tf.Tensor) -> tf.Tensor:
    def _lengths_to_indices(lengths: tf.Tensor) -> tf.RaggedTensor:
        return tf.ragged.range(
            starts=tf.cumsum(
                tf.concat([[0], lengths[:-1]], axis=0)
            ),  # Starts of index ranges
            limits=tf.cumsum(lengths),  # End of index ranges
        )

    def _cartesian_prod(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.reshape(tf.stack(tf.meshgrid(a, b, indexing="ij"), axis=-1), (-1, 2))

    indices = _lengths_to_indices(lengths)
    return tf.map_fn(lambda x: _cartesian_prod(x, x), indices).flat_values


class GraphDenselyConnectedTransformerLayer(tf.keras.layers.Layer):
    """Transformer-like layer for sparsely batched graphs.
    Applies a standard Transformer layer such that all nodes of each graph attend to
    all (other and same) nodes in the same graph. This entirely ignores the graph
    topology, and instead treats the node representations as a bag of inputs.
    """
    def __init__(
        self,
        num_heads: int = 6,
        head_size: int = 16,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        intermediate_activation_name: str = "gelu",
        hidden_dropout_rate: float = 0.1,
        attention_probs_dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_heads = num_heads
        self._head_size = head_size
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._intermediate_act_fn = get_activation_function(
            intermediate_activation_name
        )
        self._hidden_dropout_rate = hidden_dropout_rate
        self._attention_probs_dropout_rate = attention_probs_dropout_rate

        # Sublayer definitions:
        self._node_label_to_qkv_layer = tf.keras.layers.Dense(
            units=3 * (num_heads * head_size), activation=None, use_bias=False,
        )
        self._attention_to_pre_intermediate_layer = tf.keras.layers.Dense(
            units=hidden_size, activation=None, use_bias=False,
        )
        self._pre_intermediate_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self._pre_intermediate_to_intermediate_layer = tf.keras.layers.Dense(
            units=intermediate_size,
            activation=self._intermediate_act_fn,
            use_bias=False,
        )
        self._intermediate_to_output_layer = tf.keras.layers.Dense(
            units=hidden_size, activation=None, use_bias=False,
        )
        self._output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def build(self, input_shapes):
        node_embedding_shape, _ = input_shapes
        node_emb_dim = node_embedding_shape[-1]

        with tf.name_scope("node_to_qkv"):
            self._node_label_to_qkv_layer.build(tf.TensorShape((None, node_emb_dim)))
        with tf.name_scope("attention_to_pre_intermediate"):
            self._attention_to_pre_intermediate_layer.build(
                tf.TensorShape((None, self._num_heads * self._head_size))
            )
        with tf.name_scope("pre_intermediate_layer_norm"):
            self._pre_intermediate_layer_norm.build(
                tf.TensorShape((None, self._hidden_size))
            )
        with tf.name_scope("pre_intermediate_to_intermediate"):
            self._pre_intermediate_to_intermediate_layer.build(
                tf.TensorShape((None, self._hidden_size))
            )
        with tf.name_scope("intermediate_to_output_"):
            self._intermediate_to_output_layer.build(
                tf.TensorShape((None, self._intermediate_size))
            )
        with tf.name_scope("output_layer_norm"):
            self._output_layer_norm.build(tf.TensorShape((None, self._hidden_size)))

        super().build(input_shapes)

    def call(self, inputs, training: bool):
        """
        Compute Transformer layer on graph nodes.

        Used shape abbreviations:
        V ~ num _v_ertices
        D ~ vertex representation _d_imension
        H ~ attention _h_ead size
        N ~ _n_um attention heads
        P ~ _n_number of node pairs in total, where P = sum(graph_size**2 for graph_size in batch)

        Args:
            inputs: Tuple of two values:
                node_embeddings: float Tensor of shape [V, D], containing current representations
                    of all nodes.
                in_graph_node_pairs: int Tensor of shape [P, 2], essentially, the set of edges
                    if all graphs were fully connected.
            training: Flag indicating if we are in training mode (with dropout) or not.
        """
        node_embeddings, in_graph_node_pairs = inputs

        attention_output = self._per_graph_attention_layer(
            node_embeddings, in_graph_node_pairs, training=training,
        )  # [V, N * H]

        # Project back to hidden size, drop out, insert residual from input and norm:
        pre_intermediate_states = self._attention_to_pre_intermediate_layer(
            attention_output, training=training
        )

        if training:
            pre_intermediate_states = tf.nn.dropout(
                pre_intermediate_states, rate=self._hidden_dropout_rate
            )

        pre_intermediate_states = self._pre_intermediate_layer_norm(
            pre_intermediate_states + node_embeddings, training=training
        )

        # Blow up to intermediate, project back, and drop out:
        intermediate_states = self._pre_intermediate_to_intermediate_layer(
            pre_intermediate_states, training=training
        )

        layer_output = self._intermediate_to_output_layer(
            intermediate_states, training=training
        )

        if training:
            layer_output = tf.nn.dropout(
                layer_output, rate=self._hidden_dropout_rate
            )

        # Finally, insert residual from before intermediate blowup, norm and return:
        layer_output = self._output_layer_norm(
            layer_output + pre_intermediate_states, training=training
        )

        return layer_output

    def _per_graph_attention_layer(
        self,
        node_embeddings: tf.Tensor,  # [V, D], float
        in_graph_node_pairs: tf.Tensor,  # [None, 2], int
        training: bool,
    ):
        # Compute queries, keys, values in one go for all heads, then split out:
        node_qkv = self._node_label_to_qkv_layer(node_embeddings, training=training)
        node_queries, node_keys, node_values = tf.split(
            value=tf.reshape(
                node_qkv, shape=(-1, self._num_heads, 3 * self._head_size)
            ),
            num_or_size_splits=3,
            axis=-1,
        )  # 3 times [V, N, H]

        # Compute attention scores between queries and keys by doing an inner product,
        # normalised by the square root of the representation size.
        scores = tf.einsum(
            "phi,phi->ph",  # p ~ pairs, h ~ heads, i ~ representation dim index
            tf.gather(
                node_queries, indices=in_graph_node_pairs[:, 0]
            ),  # [P, N, H]
            tf.gather(
                node_keys, indices=in_graph_node_pairs[:, 1]
            ),  # [P, N, H]
        ) / tf.math.sqrt(
            tf.cast(self._head_size, tf.float32)
        )  # [P, N]

        # Compute attention weights by softmax per head, per graph:
        attention_weight_list = []
        for head_idx in range(self._num_heads):
            head_scores = scores[:, head_idx]
            head_attention_weights = unsorted_segment_softmax(
                logits=head_scores,
                segment_ids=in_graph_node_pairs[:, 0],
                num_segments=tf.shape(node_embeddings)[0],
            )
            attention_weight_list.append(head_attention_weights)
        attention_weights = tf.stack(attention_weight_list, axis=-1)  # [P, N]
        if training:
            attention_weights = tf.nn.dropout(
                attention_weights, rate=self._attention_probs_dropout_rate
            )

        # Compute new representations by doing a weighted sum using the attention weights:
        weighted_node_values = (
            tf.expand_dims(attention_weights, axis=-1)   # [P, N, 1]
            * tf.gather(node_values, in_graph_node_pairs[:, 1])  # [P, N, H]
        )  # [P, N, H]
        attention_output = tf.math.unsorted_segment_sum(
            data=weighted_node_values,  # [P, N, H]
            segment_ids=in_graph_node_pairs[:, 0],  # [P]
            num_segments=tf.shape(node_embeddings)[0],  # V
        )
        attention_output = tf.reshape(
            attention_output, shape=(-1, self._num_heads * self._head_size)
        )  # [V, N * H]

        return attention_output
