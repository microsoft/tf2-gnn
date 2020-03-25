from abc import abstractmethod
from typing import NamedTuple

import tensorflow as tf
from dpu_utils.tf2utils import MLP

from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput

from tf2_gnn.utils.gather_dense_gradient import gather_dense_gradient


class GraphGlobalExchangeInput(NamedTuple):
    """Input named tuple for graph global information exchange in GNNs."""

    node_embeddings: tf.Tensor
    node_to_graph_map: tf.Tensor
    num_graphs: tf.Tensor


class GraphGlobalExchange(tf.keras.layers.Layer):
    """Update node representations based on graph-global information."""

    def __init__(
        self,
        hidden_dim: int,
        weighting_fun: str = "softmax",
        num_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialise the layer."""
        super().__init__()
        self._hidden_dim = hidden_dim
        self._weighting_fun = weighting_fun
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def build(self, tensor_shapes: GraphGlobalExchangeInput):
        """Build the various layers in the model.

        Args:
            tensor_shapes: A GraphGlobalExchangeInput of tensor shapes.

        Returns:
            Nothing, but initialises the layers in the model based on the tensor shapes given.
        """
        self._node_to_graph_representation_layer = WeightedSumGraphRepresentation(
            graph_representation_size=self._hidden_dim,
            weighting_fun=self._weighting_fun,
            num_heads=self._num_heads,
            scoring_mlp_layers=[self._hidden_dim],
        )
        self._node_to_graph_representation_layer.build(
            NodesToGraphRepresentationInput(
                node_embeddings=tensor_shapes.node_embeddings,
                node_to_graph_map=tensor_shapes.node_to_graph_map,
                num_graphs=tensor_shapes.num_graphs,
            )
        )

        super().build(tensor_shapes)

    @abstractmethod
    def call(self, inputs: GraphGlobalExchangeInput, training: bool = False):
        """
        Args:
            inputs: A GraphGlobalExchangeInput containing the following fields:
                node_features: float32 tensor of shape [V, D], the original representation
                    of each node in the graph.
                
                node_to_graph_map: int32 tensor of shape [V], where node_to_graph_map[v] = i
                    means that node v belongs to graph i in the batch.
    
                num_graphs: int32 tensor of shape [], specifying number of graphs in batch.

            training: A bool representing whether the model is training or evaluating.

        Returns:
            A tensor of shape [V, hidden_dim]. The tensor represents the encoding of the
            states updated with information from the entire graph.
        """
        pass

    def _compute_per_node_graph_representations(
        self, inputs: GraphGlobalExchangeInput, training: bool = False
    ):
        cur_graph_representations = self._node_to_graph_representation_layer(
            NodesToGraphRepresentationInput(
                node_embeddings=inputs.node_embeddings,
                node_to_graph_map=inputs.node_to_graph_map,
                num_graphs=inputs.num_graphs,
            ),
            training=training,
        )  # Shape [G, hidden_dim]

        per_node_graph_representations = gather_dense_gradient(
            cur_graph_representations, inputs.node_to_graph_map
        )  # Shape [V, hidden_dim]

        if training:
            per_node_graph_representations = tf.nn.dropout(
                per_node_graph_representations, rate=self._dropout_rate
            )

        return per_node_graph_representations


class GraphGlobalMeanExchange(GraphGlobalExchange):
    def __init__(
        self,
        hidden_dim: int,
        weighting_fun: str = "softmax",
        num_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialise the layer."""
        super().__init__(hidden_dim, weighting_fun, num_heads, dropout_rate)

    def build(self, tensor_shapes: GraphGlobalExchangeInput):
        with tf.name_scope(self.__class__.__name__):
            super().build(tensor_shapes)

    def call(self, inputs: GraphGlobalExchangeInput, training: bool = False):
        per_node_graph_representations = self._compute_per_node_graph_representations(
            inputs, training
        )
        return (inputs.node_embeddings + per_node_graph_representations) / 2


class GraphGlobalGRUExchange(GraphGlobalExchange):
    def __init__(
        self,
        hidden_dim: int,
        weighting_fun: str = "softmax",
        num_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialise the layer."""
        super().__init__(hidden_dim, weighting_fun, num_heads, dropout_rate)

    def build(self, tensor_shapes: GraphGlobalExchangeInput):
        with tf.name_scope(self.__class__.__name__):
            self._gru_cell = tf.keras.layers.GRUCell(units=self._hidden_dim)
            self._gru_cell.build(tf.TensorShape((None, self._hidden_dim)))
            super().build(tensor_shapes)

    def call(self, inputs: GraphGlobalExchangeInput, training: bool = False):
        per_node_graph_representations = self._compute_per_node_graph_representations(
            inputs, training
        )
        cur_node_representations, _ = self._gru_cell(
            inputs=per_node_graph_representations,
            states=[inputs.node_embeddings],
            training=training,
        )
        return cur_node_representations


class GraphGlobalMLPExchange(GraphGlobalExchange):
    def __init__(
        self,
        hidden_dim: int,
        weighting_fun: str = "softmax",
        num_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialise the layer."""
        super().__init__(hidden_dim, weighting_fun, num_heads, dropout_rate)

    def build(self, tensor_shapes: GraphGlobalExchangeInput):
        with tf.name_scope(self.__class__.__name__):
            self._mlp = MLP(out_size=self._hidden_dim)
            self._mlp.build(tf.TensorShape((None, 2 * self._hidden_dim)))
            super().build(tensor_shapes)

    def call(self, inputs: GraphGlobalExchangeInput, training: bool = False):
        per_node_graph_representations = self._compute_per_node_graph_representations(
            inputs, training
        )
        cur_node_representations = self._mlp(
            tf.concat([per_node_graph_representations, inputs.node_embeddings], axis=-1),
            training=training,
        )
        return cur_node_representations
