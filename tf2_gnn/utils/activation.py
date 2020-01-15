"""Custom activation functions."""
import math as m

import tensorflow as tf


def gelu(input_tensor: tf.Tensor):
    """An approximation to the GELU activation function as used in the paper
    https://arxiv.org/pdf/1810.04805.pdf
    """
    cdf = 0.5 * (
        1.0 + tf.tanh((tf.sqrt(2 / m.pi) * (input_tensor + 0.044715 * tf.pow(input_tensor, 3))))
    )
    return input_tensor * cdf
