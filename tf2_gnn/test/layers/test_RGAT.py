"""Tests for the RGAT message passing layer."""
import tensorflow as tf
import pytest

from tf2_gnn.layers.message_passing import MessagePassingInput, RGAT


# fmt: off
shape_test_data = [
    (
        tf.TensorShape(dims=(None, 3)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(3)),
        16,
        8,
    ),
    (
        tf.TensorShape(dims=(None, 1)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(1)),
        2,
        1,
    ),
    (
        tf.TensorShape(dims=(None, 7)),
        tuple(tf.TensorShape(dims=(None, 2)) for _ in range(14)),
        64,
        4,
    ),
]
# fmt: on


@pytest.mark.parametrize(
    "node_embedding_shape,adjacency_list_shapes,hidden_dim,num_heads", shape_test_data
)
def test_rgat_layer_has_expected_number_of_trainable_variables(
    node_embedding_shape, adjacency_list_shapes, hidden_dim, num_heads
):
    # Given:
    rgat_params = RGAT.get_default_hyperparameters()
    rgat_params["hidden_dim"] = hidden_dim
    rgat_params["num_heads"] = num_heads
    rgat_layer = RGAT(rgat_params)

    # When:
    rgat_layer.build(
        MessagePassingInput(
            node_embeddings=node_embedding_shape, adjacency_lists=adjacency_list_shapes
        )
    )
    trainable_vars = rgat_layer.trainable_variables
    all_vars = rgat_layer.variables

    # Then:
    # There should be 1 dense layer and 1 attention weight per layer type
    assert len(trainable_vars) == 2 * len(adjacency_list_shapes)
    assert len(all_vars) == len(trainable_vars)  # There should be no un-trainable variables.

    for trainable_var in trainable_vars:
        if "kernel" in trainable_var.name:
            assert tuple(trainable_var.shape.as_list()) == (node_embedding_shape[-1], hidden_dim)
        elif "attention" in trainable_var.name:
            assert tuple(trainable_var.shape.as_list()) == (num_heads, 2 * hidden_dim / num_heads)
        else:
            assert False  # There should be no other trainable variable types.
