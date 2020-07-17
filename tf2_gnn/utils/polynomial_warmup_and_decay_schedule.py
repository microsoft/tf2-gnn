from typing import Union

import tensorflow as tf


class PolynomialWarmupAndDecaySchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """A LearningRateSchedule that uses a polynomial warmup and decay schedule."""

    def __init__(
        self,
        learning_rate: Union[tf.Tensor, float],
        warmup_steps: Union[tf.Tensor, int],
        decay_steps: Union[tf.Tensor, int],
        initial_learning_rate: Union[tf.Tensor, float],
        final_learning_rate: Union[tf.Tensor, float],
        power: Union[tf.Tensor, float] = 1.0,
        name=None,
    ):
        """Applies polynomial warmup and decay to the learning rate.
        In the warmup phase of the first `warmup_steps` training steps, the learning
        rate is increased from `initial_learning_rate` to `learning_rate`.
        In the following `decay_steps` steps, the learning rate is decreased to
        `final_learning_rate`.

        The schedule is a 1-arg callable that produces a decayed learning rate
        when passed the current optimizer step. This can be useful for changing the
        learning rate value across different invocations of optimizer functions.
        It is computed as:
        ```python
        def scheduled_learning_rate(step):
            if step <= warmup_steps:
                return ((learning_rate - initial_learning_rate) *
                        (step / warmup_steps)**power
                    ) + initial_learning_rate
            else:
                step = min(step - warmup_steps, decay_steps)
                return ((learning_rate - final_learning_rate) *
                        (1 - step / decay_steps)**power
                    ) + final_learning_rate
        ```

        You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
        as the learning rate.

        Args:
            learning_rate: A scalar `float32` or `float64` `Tensor` or a
                Python number.  The peak learning rate.
            warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
                Must be positive.  See the warmup computation above.
            decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
                Must be positive.  See the decay computation above.
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
                Python number, setting the initial learning rate from which we
                warm up to `learning_rate`.
            final_learning_rate: A scalar `float32` or `float64` `Tensor` or a
                Python number, setting the final learning rate to which we decay
                from `learning_rate`.
            power: A scalar `float32` or `float64` `Tensor` or a
                Python number.  The power of the polynomial. Defaults to linear, 1.0.
            name: String.  Optional name of the operation.

        Returns:
            A 1-arg callable learning rate schedule that takes the current optimizer
            step and outputs the decayed learning rate, a scalar `Tensor` of the same
            type as `initial_learning_rate`.
        """
        super(PolynomialWarmupAndDecaySchedule, self).__init__()

        self.learning_rate = learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.power = power
        self.name = name

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "power": self.power,
            "name": self.name,
        }

    def __call__(self, step):
        with tf.name_scope(self.name or "PolynomialWarmupAndDecay"):
            is_in_warmup_phase = tf.less_equal(step, self.warmup_steps)

            def _warmup_learning_rate():
                return (
                    (self.learning_rate - self.initial_learning_rate)
                    * tf.pow(step / self.warmup_steps, self.power)
                ) + self.initial_learning_rate

            def _decay_learning_rate():
                effective_step = tf.math.minimum(step - self.warmup_steps, self.decay_steps)
                return (
                    (self.learning_rate - self.final_learning_rate)
                    * tf.pow(1 - effective_step / self.decay_steps, self.power)
                ) + self.final_learning_rate

            return tf.cond(
                is_in_warmup_phase,
                true_fn=_warmup_learning_rate,
                false_fn=_decay_learning_rate,
            )
