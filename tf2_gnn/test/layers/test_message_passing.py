"""Tests for the MessagePassing layer."""
from typing import NamedTuple

import numpy.testing as nt
import pytest
import tensorflow as tf

from tf2_gnn.layers.message_passing import MessagePassingInput, MessagePassing


class PassSourceStates(MessagePassing):
    def __init__(self):
        params = super().get_default_hyperparameters()
        # Just in case defaults ever change
        params["message_activation_function"] = "relu"
        params["aggregation_function"] = "sum"
        super().__init__(params)

    def _message_function(
        self,
        edge_source_states: tf.Tensor,
        edge_target_states: tf.Tensor,
        num_incoming_to_node_per_message: tf.Tensor,
        edge_type_idx: int,
        training: bool
    ) -> tf.Tensor:
        return edge_source_states


class TestInput(NamedTuple):
    message_passing_input: MessagePassingInput
    aggregated_states: tf.Tensor


all_test_data = [
    # Node 0 state should get passed to node 1.
    TestInput(
        MessagePassingInput(
            node_embeddings=tf.constant([[1, 2, 3], [2, 4, 5]], dtype=tf.float32),
            adjacency_lists=(tf.constant([[0, 1]], dtype=tf.int32),),
        ),
        aggregated_states=tf.constant([[0, 0, 0], [1, 2, 3]], dtype=tf.float32),
    ),
    # Node 1 state should get passed to node 0, with ReLU activation zeroing out the -4.
    TestInput(
        MessagePassingInput(
            node_embeddings=tf.constant([[1, 2, 3], [2, -4, 5]], dtype=tf.float32),
            adjacency_lists=(tf.constant([[1, 0]], dtype=tf.int32),),
        ),
        aggregated_states=tf.constant([[2, 0, 5], [0, 0, 0]], dtype=tf.float32),
    ),
    # Node 0 state goes to node 1, node 1 state goes to node 0, and 0 & 1 states get aggregated in node 2.
    TestInput(
        MessagePassingInput(
            node_embeddings=tf.constant([[1, 2, 3], [2, 4, 5], [0, -7, -4]], dtype=tf.float32),
            adjacency_lists=(tf.constant([[1, 0], [0, 1], [0, 2], [1, 2]], dtype=tf.int32),),
        ),
        aggregated_states=tf.constant([[2, 4, 5], [1, 2, 3], [3, 6, 8]], dtype=tf.float32),
    ),
    # Node 2 now has a self loop in a different edge type.
    TestInput(
        MessagePassingInput(
            node_embeddings=tf.constant([[1, 2, 3], [2, 4, 5], [0, -7, -4]], dtype=tf.float32),
            adjacency_lists=(
                tf.constant([[1, 0], [0, 1], [0, 2], [1, 2]], dtype=tf.int32),
                tf.constant([[2, 2]], dtype=tf.int32),
            ),
        ),
        aggregated_states=tf.constant([[2, 4, 5], [1, 2, 3], [3, 0, 4]], dtype=tf.float32),
    ),
]


@pytest.mark.parametrize("test_data", all_test_data)
def test_message_passing_layer_aggregates_layers_as_expected(test_data: TestInput):
    # Given:
    message_passing_layer = PassSourceStates()

    # When:
    output = message_passing_layer(test_data.message_passing_input, training=False)

    # Then:
    assert output.shape == test_data.aggregated_states.shape
    nt.assert_array_almost_equal(output, test_data.aggregated_states)


@pytest.mark.parametrize("test_data", all_test_data)
def test_message_passing_layer_unchanged_under_jit_compilation(test_data: TestInput):
    # Given:
    message_passing_layer = PassSourceStates()
    message_passing_layer(test_data.message_passing_input)  # Make sure layer is built properly.
    compiled_call = tf.function(message_passing_layer.call)

    # When:
    eager_output = message_passing_layer(test_data.message_passing_input, training=False)
    jit_output = compiled_call(test_data.message_passing_input, training=False)

    # Then:
    nt.assert_array_almost_equal(eager_output, jit_output)
