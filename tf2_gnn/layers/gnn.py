"""GNN Encoder class."""
from typing import Any, Dict, NamedTuple, List, Tuple, Optional

import tensorflow as tf

from tf2_gnn.utils.param_helpers import get_activation_function
from .message_passing import (
    MessagePassing,
    MessagePassingInput,
    get_message_passing_class,
)
from .graph_global_exchange import (
    GraphGlobalExchangeInput,
    GraphGlobalExchange,
    GraphGlobalMeanExchange,
    GraphGlobalGRUExchange,
    GraphGlobalMLPExchange,
)


class GNNInput(NamedTuple):
    """Input named tuple for the GNN."""

    node_features: tf.Tensor
    adjacency_lists: Tuple[tf.Tensor, ...]
    node_to_graph_map: tf.Tensor
    num_graphs: tf.Tensor


class GNN(tf.keras.layers.Layer):
    """Encode graph states using a combination of graph message passing layers and dense layers

    Example usage:
    >>> layer_input = GNNInput(
    ...     node_features = tf.random.normal(shape=(5, 3)),
    ...     adjacency_lists = (
    ...         tf.constant([[0, 1], [1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
    ...         tf.constant([[2, 0]], dtype=tf.int32)
    ...         ),
    ...     node_to_graph_map = tf.fill(dims=(5,), value=0),
    ...     num_graphs = 1,
    ...     )
    ...
    >>> params = GNN.get_default_hyperparameters()
    >>> params["hidden_dim"] = 12
    >>> layer = GNN(params)
    >>> output = layer(layer_input)
    >>> print(output)
    tf.Tensor(..., shape=(5, 12), dtype=float32)
    """

    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        """Get the default hyperparameter dictionary for the  class."""
        these_hypers = {
            "message_calculation_class": "rgcn",
            "initial_node_representation_activation": "tanh",
            "dense_intermediate_layer_activation": "tanh",
            "num_layers": 4,
            "dense_every_num_layers": 2,
            "residual_every_num_layers": 2,
            "use_inter_layer_layernorm": False,
            "hidden_dim": 16,
            "layer_input_dropout_rate": 0.0,
            "global_exchange_mode": "gru",  # One of "mean", "mlp", "gru"
            "global_exchange_every_num_layers": 2,
            "global_exchange_weighting_fun": "softmax",  # One of "softmax", "sigmoid"
            "global_exchange_num_heads": 4,
            "global_exchange_dropout_rate": 0.2,
        }  # type: Dict[str, Any]
        if mp_style is not None:
            these_hypers["message_calculation_class"] = mp_style
        message_passing_class = get_message_passing_class(
            these_hypers["message_calculation_class"]
        )
        message_passing_hypers = message_passing_class.get_default_hyperparameters()
        message_passing_hypers.update(these_hypers)
        return message_passing_hypers

    def __init__(self, params: Dict[str, Any]):
        """Initialise the layer."""
        super().__init__()
        self._params = params
        self._hidden_dim = params["hidden_dim"]
        self._num_layers = params["num_layers"]
        self._dense_every_num_layers = params["dense_every_num_layers"]
        self._residual_every_num_layers = params["residual_every_num_layers"]
        self._use_inter_layer_layernorm = params["use_inter_layer_layernorm"]
        self._initial_node_representation_activation_fn = get_activation_function(
            params["initial_node_representation_activation"]
        )
        self._dense_intermediate_layer_activation_fn = get_activation_function(
            params["dense_intermediate_layer_activation"]
        )
        self._message_passing_class = get_message_passing_class(
            params["message_calculation_class"]
        )

        if not params["global_exchange_mode"].lower() in {"mean", "mlp", "gru"}:
            raise ValueError(
                f"Unknown global_exchange_mode mode {params['global_exchange_mode']} - has to be one of 'mean', 'mlp', 'gru'!"
            )
        self._global_exchange_mode = params["global_exchange_mode"]
        self._global_exchange_every_num_layers = params["global_exchange_every_num_layers"]
        self._global_exchange_weighting_fun = params["global_exchange_weighting_fun"]
        self._global_exchange_num_heads = params["global_exchange_num_heads"]
        self._global_exchange_dropout_rate = params["global_exchange_dropout_rate"]

        # Layer member variables. To be filled in in the `build` method.
        self._initial_projection_layer: tf.keras.layers.Layer = None
        self._mp_layers: List[MessagePassing] = []
        self._inter_layer_layernorms: List[tf.keras.layers.Layer] = []
        self._dense_layers: Dict[str, tf.keras.layers.Layer] = {}
        self._global_exchange_layers: Dict[str, GraphGlobalExchange] = {}

    def build(self, tensor_shapes: GNNInput):
        """Build the various layers in the model.

        Args:
            tensor_shapes: A GNNInput of tensor shapes.

        Returns:
            Nothing, but initialises the layers in the model based on the tensor shapes given.
        """
        # First, we go through the input shapes and make sure that anything which might vary batch
        # to batch (number of nodes / number of edges) is set to None.
        initial_node_features_shape: tf.TensorShape = tensor_shapes.node_features
        variable_node_features_shape = tf.TensorShape((None, initial_node_features_shape[1]))
        adjacency_list_shapes = tensor_shapes.adjacency_lists
        embedded_shape = tf.TensorShape((None, self._hidden_dim))

        with tf.name_scope(f"{self._message_passing_class.__name__}_GNN"):
            # Then we construct the layers themselves:
            with tf.name_scope("gnn_initial_node_projection"):
                self._initial_projection_layer = tf.keras.layers.Dense(
                    units=self._hidden_dim,
                    use_bias=False,
                    activation=self._initial_node_representation_activation_fn,
                )
                self._initial_projection_layer.build(variable_node_features_shape)

            # Construct the graph message passing layers.
            for layer_idx in range(self._num_layers):
                with tf.name_scope(f"Layer_{layer_idx}"):
                    with tf.name_scope("MessagePassing"):
                        self._mp_layers.append(
                            self._message_passing_class(self._params)
                        )
                        self._mp_layers[-1].build(
                            MessagePassingInput(embedded_shape, adjacency_list_shapes)
                        )

                    # If required, prepare for a LayerNorm:
                    if self._use_inter_layer_layernorm:
                        with tf.name_scope(f"LayerNorm"):
                            self._inter_layer_layernorms.append(
                                tf.keras.layers.LayerNormalization()
                            )
                            self._inter_layer_layernorms[-1].build(embedded_shape)

                    # Construct the per-node dense layers.
                    if layer_idx % self._dense_every_num_layers == 0:
                        with tf.name_scope(f"Dense"):
                            self._dense_layers[str(layer_idx)] = tf.keras.layers.Dense(
                                units=self._hidden_dim,
                                use_bias=False,
                                activation=self._dense_intermediate_layer_activation_fn,
                            )
                            self._dense_layers[str(layer_idx)].build(embedded_shape)

                    if (
                        layer_idx
                        and layer_idx % self._global_exchange_every_num_layers == 0
                    ):
                        with tf.name_scope(f"Global_Exchange"):
                            if self._global_exchange_mode.lower() == "mean":
                                exchange_layer_class = GraphGlobalMeanExchange
                            elif self._global_exchange_mode.lower() == "gru":
                                exchange_layer_class = GraphGlobalGRUExchange
                            elif self._global_exchange_mode.lower() == "mlp":
                                exchange_layer_class = GraphGlobalMLPExchange
                            exchange_layer = exchange_layer_class(
                                hidden_dim=self._hidden_dim,
                                weighting_fun=self._global_exchange_weighting_fun,
                                num_heads=self._global_exchange_num_heads,
                                dropout_rate=self._global_exchange_dropout_rate,
                            )
                            exchange_layer.build(
                                GraphGlobalExchangeInput(
                                    node_embeddings=tf.TensorShape(
                                        (None, self._hidden_dim)
                                    ),
                                    node_to_graph_map=tf.TensorShape((None,)),
                                    num_graphs=tf.TensorShape(()),
                                )
                            )
                            self._global_exchange_layers[
                                str(layer_idx)
                            ] = exchange_layer

        super().build(tensor_shapes)

        # The following is needed to work around a limitation in the @tf.function annotation.
        # (See https://github.com/tensorflow/tensorflow/issues/32457 for a related issue,
        #  though there are many more).
        # Our aim is to trace the `call` function once and for all. However, as the first
        # dimension of node features and adjacency lists keeps changing between batches (with
        # the number of nodes/edges in the batch), generalisation doesn't work automatically.
        # Instead, we have to specify the input spec explicitly; but as this depends on a
        # build-time constant (the number of edges), we cannot do that by just using @tf.function.
        # Instead, we construct the TensorSpec explicitly, and then use setattr to wrap
        # our function using tf.function.
        #
        # Finally, the `return_all_representations` option changes the shape of the return values,
        # but a tf.function-traced function must return the same shape on all code paths. To
        # handle this, we let the core function _always_ return all representations (and trace
        # that for performance reasons), and then use a thin wrapper `call` function to drop
        # the unneeded return value if needed.
        internal_call_input_spec = (
            GNNInput(
                node_features=tf.TensorSpec(shape=variable_node_features_shape, dtype=tf.float32),
                adjacency_lists=tuple(
                    tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
                    for _ in range(len(adjacency_list_shapes))
                ),
                node_to_graph_map=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                num_graphs=tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.bool)
        )
        setattr(self, "_internal_call", tf.function(func=self._internal_call, input_signature=internal_call_input_spec))

    def call(self, inputs: GNNInput, training: bool = False, return_all_representations: bool = False):
        """
        Args:
            inputs: A GNNInput containing the following fields:
                node_features: float32 tensor of shape [V, D], the original representation
                    of each node in the graph.

                adjacency_lists: an tuple of tensors of shape [E, 2] which represents an adjacency
                    list for a given edge type. Concretely,
                        adjacency_list[l][k,:] == [v, u]
                    means that the k-th edge of type l connects node v to node u.

                node_to_graph_map: int32 tensor of shape [V], where node_to_graph_map[v] = i
                    means that node v belongs to graph i in the batch.

                num_graphs: int32 tensor of shape [], specifying number of graphs in batch.

            training: A bool representing whether the model is training or evaluating.

            return_all_representations: A bool indicating whether to return all (initial,
                intermediate, and final) GNN results as well.

        Returns:
            If return_all_representations is False (the default):
            A tensor of shape [V, hidden_dim], where hidden_dim was defined in the layer
            initialisation. The tensor represents the encoding of the initial node_features by the
            GNN framework.

            If return_all_representations is True:
            A pair, first element as for return_all_representations=False, second element a  list
            of Tensors of shape [V, hidden_dim], where the first element is the original GNN
            input (after a potential projection layer) and the remaining elements are the
            output of all GNN layers (without dropout, residual connections, dense layers 
            or layer norm applied).
        """
        cur_node_representations, all_node_representations = self._internal_call(inputs, training)

        if return_all_representations:
            return cur_node_representations, all_node_representations

        return cur_node_representations

    def _internal_call(self, inputs: GNNInput, training: bool = False):
        initial_node_features: tf.Tensor = inputs.node_features
        adjacency_lists = inputs.adjacency_lists
        cur_node_representations = self._initial_projection_layer(initial_node_features)

        # Layer loop.
        last_node_representations = cur_node_representations
        all_node_representations = [cur_node_representations]
        for layer_idx, mp_layer in enumerate(self._mp_layers):
            if training:
                cur_node_representations = tf.nn.dropout(
                    cur_node_representations, rate=self._params["layer_input_dropout_rate"]
                )

            # Pass residuals through:
            if layer_idx % self._residual_every_num_layers == 0:
                tmp = cur_node_representations
                if layer_idx > 0:
                    cur_node_representations += last_node_representations
                    cur_node_representations /= 2
                last_node_representations = tmp

            # Apply this message passing layer.
            cur_node_representations = mp_layer(
                MessagePassingInput(
                    node_embeddings=cur_node_representations, adjacency_lists=adjacency_lists
                ),
                training=training,
            )
            all_node_representations.append(cur_node_representations)

            if layer_idx and layer_idx % self._global_exchange_every_num_layers == 0:
                cur_node_representations = self._global_exchange_layers[str(layer_idx)](
                    GraphGlobalExchangeInput(
                        node_embeddings=cur_node_representations,
                        node_to_graph_map=inputs.node_to_graph_map,
                        num_graphs=inputs.num_graphs,
                    ),
                    training=training,
                )

            # If required, apply a LayerNorm:
            if self._use_inter_layer_layernorm:
                cur_node_representations = self._inter_layer_layernorms[layer_idx](
                    cur_node_representations
                )

            # Apply dense layer, if needed.
            if layer_idx % self._dense_every_num_layers == 0:
                cur_node_representations = self._dense_layers[str(layer_idx)](
                    cur_node_representations, training=training
                )

        return cur_node_representations, tuple(all_node_representations)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
