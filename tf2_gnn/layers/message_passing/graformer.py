from typing import Dict, List, Any, Union

import tensorflow as tf
from dpu_utils.tf2utils import unsorted_segment_log_softmax

from tf2_gnn.layers.message_passing.message_passing import (
    MessagePassingInput,
    register_message_passing_implementation,
)
from tf2_gnn.utils.param_helpers import get_activation_function


@register_message_passing_implementation
class Graformer(tf.keras.layers.Layer):
    """Compute new graph states by a mix of neural message passing and (self-) attention.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell, edge features E_{u,\ell,v}.

    We compute new states as follows (at timestep t, for head h):
        m^{t,h}_{u,\ell,v} = V^{t,h}_\ell * h^t_u
        s^{t,h}_{u,\ell,v} = (K^{t,h}_\ell * h^t_u + bm^{t,h}_\ell + F^{t,h}_\ell * E_{u,\ell,v})^T * (Q^{t,h}_\ell * h^t_v) + ba^{t,h}_\ell
        a^{t,h}_{:,:,v} = softmax(s^{t,h}_{u,\ell,v})
        h^{t+1}_v =
            LayerNorm(
                h^t_v
                + Dropout(
                    P^t * ||_h \sum_{u,\ell} (a^{t,h}_{u,\ell,v} * m^{t,h}_{u,\ell,v})
                  ))
    The learnable parameters of this are the following:
        V^{t,h}_\ell (the value projection of the node state for head h at step t)
        K^{t,h}_\ell (the key projection of the node state for head h at step t)
        Q^{t,h}_\ell (the query projection of the node state for head h at step t)
        bm^{t,h}_\ell (the multiplicative edge bias for edge type \ell at step t)
        ba^{t,h}_\ell (the additive edge bias for edge type \ell at step t)
        F^{t,h}_\ell (the edge feature projection for type \ell at step t)
        P^t (the head combination projection at step t)

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * L: number of different edge types
    * E: number of edges of a given edge type
    * M: number of messages (sum of E for all edge types)
    * H: output node representation dimension

    >>> node_embeddings = tf.random.normal(shape=(5, 12))
    >>> adjacency_lists = (
    ...    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[2, 3], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[3, 1]], dtype=tf.int32),
    ... )
    ...
    >>> edge_features = (
    ...    tf.zeros((3, 0), dtype=tf.float32),  # No edge feats
    ...    tf.random.normal(shape=(2, 4)),
    ...    tf.random.normal(shape=(1, 2)),
    ... )
    ...
    >>> params = Graformer.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = Graformer(params)
    >>> output = layer(MessagePassingInput(node_embeddings, adjacency_lists, edge_features))
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            "message_activation_function": "gelu",  # One of relu, leaky_relu, elu, gelu, tanh
            "hidden_dim": 7,
            "num_heads": 4,
            "msg_dim": 5,
            "keyquery_dim": 3,
            "dropout_rate": 0.2,
            "use_ff_sublayer": True,
            "rezero_mode": "vector",  # one of ["off", "scalar", "vector"], see https://arxiv.org/pdf/2003.04887.pdf and https://arxiv.org/pdf/2103.17239.pdf.
        }

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)

        activation_fn_name = params["message_activation_function"]
        self._activation_fn = get_activation_function(activation_fn_name)

        self._hidden_dim = params["hidden_dim"]
        self._num_heads = params["num_heads"]
        self._msg_dim = params["msg_dim"]
        self._keyquery_dim = params["keyquery_dim"]
        self._dropout_rate = params["dropout_rate"]
        self._use_ff_sublayer = params["use_ff_sublayer"]
        self._rezero_mode = params["rezero_mode"]

    def build(self, input_shapes: MessagePassingInput):
        self._num_edge_types = len(input_shapes.adjacency_lists)
        node_in_size = input_shapes.node_embeddings[-1]

        assert (
            node_in_size == self._hidden_dim
        ), "Graformer requires input = output hidden dim."

        # For efficiency reasons, we compute the key/query/value projections for all heads
        # and edge types in one go:
        single_head_kqv_size = 2 * self._keyquery_dim + self._msg_dim
        kqv_size = self._num_heads * single_head_kqv_size
        # TODO(mabrocks): thinkg about allow slicing to implement towers-like structure
        with tf.name_scope("kqv"):
            self._kqv_proj = tf.keras.layers.Dense(
                units=self._num_edge_types * kqv_size, use_bias=False, activation=None
            )
            self._kqv_proj.build(tf.TensorShape((None, self._hidden_dim)))

        self._edge_to_mult_type_bias: List[tf.Variable] = []
        self._edge_to_add_type_bias: List[tf.Variable] = []
        self._edge_to_edgefeat_proj: Dict[int, tf.keras.layers.Dense] = {}
        for edge_type_idx in range(self._num_edge_types):
            with tf.name_scope(f"edge_type_{edge_type_idx}"):
                if input_shapes.edge_features[edge_type_idx][-1] > 0:
                    with tf.name_scope("edge_feat_proj"):
                        edge_feat_proj = tf.keras.layers.Dense(
                            units=self._num_heads * self._keyquery_dim,
                            use_bias=False,
                            activation=None,
                        )
                        edge_feat_proj.build(input_shapes.edge_features[edge_type_idx])
                        self._edge_to_edgefeat_proj[edge_type_idx] = edge_feat_proj
                self._edge_to_mult_type_bias.append(
                    self.add_weight(
                        name="type_multiplicative_bias",
                        shape=(self._num_heads, self._keyquery_dim),
                        trainable=True,
                        initializer="zeros",  # this means to "has no effect"
                    )
                )
                self._edge_to_add_type_bias.append(
                    self.add_weight(
                        name="type_additive_bias",
                        shape=(self._num_heads,),
                        trainable=True,
                        initializer="zeros",  # this means to "has no effect"
                    )
                )

        with tf.name_scope("combination_proj"):
            self._combination_proj = tf.keras.layers.Dense(
                units=self._hidden_dim, use_bias=False, activation=None
            )
            self._combination_proj.build(
                tf.TensorShape((None, self._num_heads * self._msg_dim))
            )

        with tf.name_scope("pre_mp"):
            self._pre_mp_norm = tf.keras.layers.LayerNormalization()
            self._pre_mp_norm.build(tf.TensorShape((None, self._hidden_dim)))

        with tf.name_scope("post_mp"):
            self._post_mp_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
            self._post_mp_dropout.build(tf.TensorShape((None, self._hidden_dim)))

        with tf.name_scope("rezero_parameters"):
            if self._rezero_mode == "off":
                self._rezero_scaler: Union[float, tf.Tensor] = 1.0
            elif self._rezero_mode == "scalar":
                self._rezero_scaler = self.add_weight(
                    name="rezero_scaler", shape=(1,), initializer="zeros"
                )
            elif self._rezero_mode == "vector":
                self._rezero_scaler = self.add_weight(
                    name="rezero_scaler", shape=(self._hidden_dim,), initializer="zeros"
                )
            else:
                raise ValueError(f"Unrecognized rezero mode `{self._rezero_mode}`.")

        if self._use_ff_sublayer:
            with tf.name_scope("ff_sublayer"):
                with tf.name_scope("pre_norm"):
                    self._pre_ff_norm = tf.keras.layers.LayerNormalization()
                    self._pre_ff_norm.build(tf.TensorShape((None, self._hidden_dim)))
                with tf.name_scope("ff_layer1"):
                    self._ff_layer1 = tf.keras.layers.Dense(
                        units=2 * self._hidden_dim,
                        use_bias=False,
                        activation=self._activation_fn,
                    )
                    self._ff_layer1.build(tf.TensorShape((None, self._hidden_dim)))
                with tf.name_scope("ff_layer2"):
                    self._ff_layer2 = tf.keras.layers.Dense(
                        units=self._hidden_dim, use_bias=False, activation=None
                    )
                    self._ff_layer2.build(tf.TensorShape((None, 2 * self._hidden_dim)))
                self._post_ff_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
                self._post_ff_dropout.build(tf.TensorShape((None, self._hidden_dim)))

                with tf.name_scope("rezero_parameters"):
                    if self._rezero_mode == "off":
                        self._post_ff_rezero_scaler: Union[float, tf.Tensor] = 1.0
                    elif self._rezero_mode == "scalar":
                        self._post_ff_rezero_scaler = self.add_weight(
                            name="rezero_scaler", shape=(1,), initializer="zeros"
                        )
                    elif self._rezero_mode == "vector":
                        self._post_ff_rezero_scaler = self.add_weight(
                            name="rezero_scaler",
                            shape=(self._hidden_dim,),
                            initializer="zeros",
                        )
                    else:
                        raise ValueError(f"Unrecognized rezero mode `{self._rezero_mode}`.")

        super().build(input_shapes)

    def call(self, inputs: MessagePassingInput, training: bool = False):
        """Call the message passing layer.

        Args:
            inputs: A tuple containing three items:
                node_embeddings: float32 tensor of shape [V, D], the original representation of each
                    node in the graph.
                adjacency_lists: Tuple of L adjacency lists, represented as int32 tensors of shape
                    [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
                    of type l connects node v to node u.
                edge_features: Tuple of L edge feature arrays of shape [E, ED]. Concretely,
                    edge_features[l][k] = f means that the k-th edge of type l has features f.
            training: A bool that denotes whether we are in training mode.

        Returns:
            float32 tensor of shape [V, hidden_dim]
        """
        num_nodes = tf.shape(inputs.node_embeddings)[0]
        normed_node_embs = self._pre_mp_norm(inputs.node_embeddings)

        # For efficiency reasons, we compute the key/query/value projections for all heads
        # and edge types in one go:
        node_kqv = self._kqv_proj(
            normed_node_embs
        )  # [V, L * num_heads * (2 * keyquery_dim + msg_dim)]
        node_keys, node_queries, node_values = tf.split(
            value=tf.reshape(
                node_kqv,
                (
                    num_nodes,
                    self._num_edge_types,
                    self._num_heads,
                    2 * self._keyquery_dim + self._msg_dim,
                ),
            ),
            num_or_size_splits=[self._keyquery_dim, self._keyquery_dim, self._msg_dim],
            axis=-1,
        )  # each of shape [V, L, num_heads, keyquery_dim/msg_dim]

        all_msgs: List[
            tf.Tensor
        ] = []  # num_edge_types elements of shape [E, num_heads, msg_dim]
        all_scores: List[
            List[tf.Tensor]
        ] = []  # num_edge_types elements of shape [E, num_heads]
        all_targets: List[tf.Tensor] = []  # num_edge_type elements of shape [E]
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(
            inputs.adjacency_lists
        ):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]

            # = Compute (unweighted) messages:
            # m^{t,h}_{u,\ell,v} = V^{t,h}_\ell * h^t_u  (all heads for fixed \ell in one go):
            edge_msgs = tf.nn.embedding_lookup(
                params=node_values[:, edge_type_idx, :, :], ids=edge_sources
            )  # [E, num_heads, msg_dim]

            # = Compute attention scores:
            # F^{t,h}_\ell * E_{u,\ell,v}:
            edgefeat_proj = self._edge_to_edgefeat_proj.get(edge_type_idx)
            if edgefeat_proj is not None:
                edge_feat_bias = tf.reshape(
                    edgefeat_proj(inputs.edge_features[edge_type_idx]),
                    (-1, self._num_heads, self._keyquery_dim),
                )  # [E, num_heads, keyquery_dim]
            else:
                edge_feat_bias = None
            edge_queries = tf.nn.embedding_lookup(
                params=node_queries[:, edge_type_idx, :, :], ids=edge_targets
            )  # [E, num_heads, keyquery_dim]

            # s^{t,h}_{u,\ell,v} = (K^{t,h}_\ell * h^t_u + bm^{t,h}_\ell + F^{t,h}_\ell * E_{u,\ell,v})^T * (Q^{t,h}_\ell * h^t_v) + ba^{t,h}_\ell
            edge_keys = tf.nn.embedding_lookup(
                params=node_keys[:, edge_type_idx, :, :], ids=edge_sources
            )  # [E, num_heads, keyquery_dim]

            # bm^{t,h}_\ell and ba^{t,h}_\ell
            edge_mult_type_bias = self._edge_to_mult_type_bias[
                edge_type_idx
            ]  # [num_heads, keyquery_dim]
            if edge_feat_bias is not None:
                keys = (
                    edge_keys + tf.expand_dims(edge_mult_type_bias, 0) + edge_feat_bias
                )
            else:
                keys = edge_keys + tf.expand_dims(edge_mult_type_bias, 0)
            edge_scores = tf.einsum("vhk,vhk->vh", keys, edge_queries)  # [E, num_heads]
            edge_scores = edge_scores + tf.expand_dims(
                self._edge_to_add_type_bias[edge_type_idx], 0
            )

            all_msgs.append(edge_msgs)
            all_scores.append(edge_scores)
            all_targets.append(edge_targets)

        msgs = tf.concat(all_msgs, axis=0)  # [M, num_heads, msg_dim]
        scores = tf.concat(all_scores, axis=0)  # [M, num_heads]
        # Normalise scores by sqrt of Key/Query dim:
        scores = scores / tf.sqrt(tf.constant(self._keyquery_dim, dtype=tf.float32))
        targets = tf.concat(all_targets, axis=0)  # [M]

        head_to_aggregated_messages = []  # num_heads tensors of shape [V, msg_dim]
        for head_idx in range(self._num_heads):
            # a^{t,h}_{:,:,v} = softmax(s^{t,h}_{u,\ell,v})
            head_scores = scores[:, head_idx]  # [M]
            head_attention = tf.exp(
                unsorted_segment_log_softmax(
                    logits=head_scores, segment_ids=targets, num_segments=num_nodes
                )
            )  # [M]
            head_msgs = msgs[:, head_idx, :]  # [M, msg_dim]
            # Compute weighted sum per target node for this head:
            head_to_aggregated_messages.append(
                tf.math.unsorted_segment_sum(
                    data=tf.expand_dims(head_attention, -1) * head_msgs,
                    segment_ids=targets,
                    num_segments=num_nodes,
                )
            )

        aggregated_messages = tf.concat(
            head_to_aggregated_messages, axis=1
        )  # [V, num_heads * msg_dim]
        proj_aggregated_messages = self._combination_proj(aggregated_messages)  # [V, H]

        new_node_states = (
            inputs.node_embeddings
            + self._rezero_scaler * self._post_mp_dropout(proj_aggregated_messages)
        )

        if self._use_ff_sublayer:
            ff_input = self._pre_ff_norm(new_node_states)
            ff_output = self._post_ff_dropout(
                self._ff_layer2(self._ff_layer1(ff_input))
            )
            new_node_states = new_node_states + self._post_ff_rezero_scaler * ff_output

        return new_node_states


if __name__ == "__main__":
    import doctest
    import pdb
    import traceback

    try:
        doctest.testmod(raise_on_error=True, optionflags=doctest.ELLIPSIS)
    except doctest.UnexpectedException as e:
        _, __, tb = e.exc_info
        traceback.print_exception(*e.exc_info)
        pdb.post_mortem(tb)
