"""Tests for the RGCN message passing layer."""
import tensorflow as tf
import pytest

from tf2_gnn.layers.message_passing import MessagePassingInput, RGCN


shape_test_data = [
    (tf.TensorShape(dims=(None, 3)), tuple(tf.TensorShape(dims=(None, 2)) for _ in range(3)), 5),
    (tf.TensorShape(dims=(None, 1)), tuple(tf.TensorShape(dims=(None, 2)) for _ in range(1)), 1),
    (tf.TensorShape(dims=(None, 7)), tuple(tf.TensorShape(dims=(None, 2)) for _ in range(14)), 7),
]


@pytest.mark.parametrize("node_embedding_shape,adjacency_list_shapes,hidden_dim", shape_test_data)
def test_rgcn_layer_has_expected_number_of_trainable_variables_when_not_using_source_and_target(
    node_embedding_shape, adjacency_list_shapes, hidden_dim
):
    # Given:
    rgcn_params = RGCN.get_default_hyperparameters()
    rgcn_params["hidden_dim"] = hidden_dim
    rgcn_params["use_target_state_as_input"] = False
    rgcn_layer = RGCN(rgcn_params)

    # When:
    rgcn_layer.build(
        MessagePassingInput(
            node_embeddings=node_embedding_shape, adjacency_lists=adjacency_list_shapes
        )
    )
    trainable_vars = rgcn_layer.trainable_variables
    all_vars = rgcn_layer.variables

    # Then:
    assert len(trainable_vars) == len(adjacency_list_shapes)  # One dense layer per layer type.
    assert len(all_vars) == len(trainable_vars)  # There should be no un-trainable variables.

    for trainable_var in trainable_vars:
        assert tuple(trainable_var.shape.as_list()) == (node_embedding_shape[-1], hidden_dim)


@pytest.mark.parametrize("node_embedding_shape,adjacency_list_shapes,hidden_dim", shape_test_data)
def test_rgcn_layer_has_expected_number_of_trainable_variables_when_using_source_and_target(
    node_embedding_shape, adjacency_list_shapes, hidden_dim
):
    # Given:
    rgcn_params = RGCN.get_default_hyperparameters()
    rgcn_params["hidden_dim"] = hidden_dim
    rgcn_params["use_target_state_as_input"] = True
    rgcn_layer = RGCN(rgcn_params)

    # When:
    rgcn_layer.build(
        MessagePassingInput(
            node_embeddings=node_embedding_shape, adjacency_lists=adjacency_list_shapes
        )
    )
    trainable_vars = rgcn_layer.trainable_variables
    all_vars = rgcn_layer.variables

    # Then:
    assert len(trainable_vars) == len(adjacency_list_shapes)  # One dense layer per layer type.
    assert len(all_vars) == len(trainable_vars)  # There should be no un-trainable variables.
    for trainable_var in trainable_vars:
        assert tuple(trainable_var.shape.as_list()) == (2 * node_embedding_shape[-1], hidden_dim)
