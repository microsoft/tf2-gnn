from typing import NamedTuple, Optional

import tensorflow as tf
from tensorflow.python.keras import initializers

from tf2_gnn.utils.constants import SMALL_NUMBER


class GraphNormInput(NamedTuple):
    """Input named tuple for the GraphNorm."""

    node_features: tf.Tensor
    node_to_graph_map: tf.Tensor


class GraphNorm(tf.keras.layers.Layer):
    """Implementation of Graph Norm (https://arxiv.org/pdf/2009.03294.pdf).
    Normalises node representations by the graph mean/variance.
    Given node representations h_{i, j} from a single graph, computes
      GraphNorm(h_{i, j}) = \gamma_j * (h_{i, j} - \alpha_j * \mu_j) / \sigma_j + \beta_j
    with \alpha_j, \beta_j, \gamma_j learnable, and 
      \mu_j      = 1/n \sum_i^n h_{i, j}
      \sigma_j^2 = 1/n \sum_i^n (h_{i, j} - \alpha_j * \mu_j)^2
    """
    def __init__(
        self,
        center: bool = True,
        scale: bool = True,
        learnable_shift: bool=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._center = center
        self._scale = scale
        self._learnable_shift = learnable_shift

    def build(self, input_shape: GraphNormInput):
        params_shape = (input_shape.node_features[-1],)

        if self._learnable_shift:
            self.alpha = self.add_weight(
                name='alpha',
                shape=params_shape,
                initializer=initializers.get('ones'),
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self.alpha = None

        if self._center:
            self.beta = self.add_weight(
                name='beta',
                shape=params_shape,
                initializer=initializers.get('zero'),
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self.beta = None

        if self._scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=params_shape,
                initializer=initializers.get('ones'),
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self.gamma = None

        super().build(input_shape)

    def call(self, inputs: GraphNormInput, training: Optional[bool]=None):
        # Compute mean
        graph_means = tf.math.segment_mean(
            data=inputs.node_features, segment_ids=inputs.node_to_graph_map
        )  # Shape [G, GD]

        per_node_graph_means = tf.gather(
            params=graph_means,
            indices=inputs.node_to_graph_map,
        )

        if self._learnable_shift:
            centered_node_features = inputs.node_features - self.alpha * per_node_graph_means
        else:
            centered_node_features = inputs.node_features - per_node_graph_means

        graph_variances = tf.math.segment_mean(
            data=tf.square(centered_node_features),
            segment_ids=inputs.node_to_graph_map,
        )  # Shape [G, GD])

        per_node_graph_stdev = tf.gather(
            params=tf.sqrt(graph_variances),
            indices=inputs.node_to_graph_map,
        )

        output = centered_node_features / (per_node_graph_stdev + SMALL_NUMBER)

        if self._scale:
            output *= self.gamma

        if self._center:
            output += self.beta

        return output
