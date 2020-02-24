"""Functions to convert from string parameters to their values."""
import tensorflow as tf

from .activation import gelu


def get_aggregation_function(aggregation_fn_name: str):
    """Convert from an aggregation function name to the function itself."""
    string_to_aggregation_fn = {
        "sum": tf.math.unsorted_segment_sum,
        "max": tf.math.unsorted_segment_max,
        "mean": tf.math.unsorted_segment_mean,
        "sqrt_n": tf.math.unsorted_segment_sqrt_n,
    }
    aggregation_fn = string_to_aggregation_fn.get(aggregation_fn_name)
    if aggregation_fn is None:
        raise ValueError(f"Unknown aggregation function: {aggregation_fn_name}")
    return aggregation_fn


def get_activation_function(activation_fn_name: str):
    """Convert from an activation function name to the function itself."""
    if activation_fn_name is None:
        return None
    activation_fn_name = activation_fn_name.lower()

    string_to_activation_fn = {
        "linear": None,
        "tanh": tf.nn.tanh,
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "gelu": gelu,
    }
    activation_fn = string_to_activation_fn.get(activation_fn_name)
    if activation_fn is None:
        raise ValueError(f"Unknown activation function: {activation_fn_name}")
    return activation_fn
