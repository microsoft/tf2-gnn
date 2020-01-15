import tensorflow as tf



# TODO: This is a band-aid around unimplemented support for sparse gradients in TF2.0
# We only use it in a situation where the gradients are always covering all inputs
# (and hence sparse/dense has little difference), so this is not a substantial
# performance hit, just inconvenient code.
@tf.custom_gradient
def gather_dense_gradient(params, indices):
    def grad(ys):
        return tf.scatter_nd(tf.expand_dims(indices, 1), ys, tf.shape(params)), None

    return tf.gather(params, indices), grad