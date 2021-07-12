import time
from abc import abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Iterable, Union

import tensorflow as tf

from tf2_gnn import GNNInput, GNN
from tf2_gnn.data import GraphDataset
from tf2_gnn.utils.polynomial_warmup_and_decay_schedule import (
    PolynomialWarmupAndDecaySchedule,
)


class GraphTaskModel(tf.keras.Model):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        """Get the default hyperparameter dictionary for the class."""
        params = {f"gnn_{name}": value for name, value in GNN.get_default_hyperparameters(mp_style).items()}
        these_hypers: Dict[str, Any] = {
            "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
            "learning_rate": 0.001,
            "learning_rate_warmup_steps": None,
            "learning_rate_decay_steps": None,
            "momentum": 0.85,
            "rmsprop_rho": 0.98,  # decay of gradients in RMSProp (unused otherwise)
            "gradient_clip_value": None,  # Set to float value to clip each gradient separately
            "gradient_clip_norm": None,  # Set to value to clip gradients by their norm
            "gradient_clip_global_norm": None,  # Set to value to clip gradients by their global norm
            "use_intermediate_gnn_results": False,
        }
        params.update(these_hypers)
        return params

    def __init__(
        self,
        params: Dict[str, Any],
        dataset: GraphDataset,
        name: str = None,
        disable_tf_function_build: bool = False
    ):
        super().__init__(name=name)
        self._params = params
        self._num_edge_types = dataset.num_edge_types
        self._use_intermediate_gnn_results = params.get(
            "use_intermediate_gnn_results", False
        )
        self._disable_tf_function_build = disable_tf_function_build

        # Keep track of the training step as a TF variable
        self._train_step_counter = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int32, name="training_step"
        )

        # Store a couple of descriptions for jit compilation in the build function.
        batch_description = dataset.get_batch_tf_data_description()
        self._batch_feature_names = tuple(batch_description.batch_features_types.keys())
        self._batch_label_names = tuple(batch_description.batch_labels_types.keys())
        self._batch_feature_spec = tuple(
            tf.TensorSpec(
                shape=batch_description.batch_features_shapes[name],
                dtype=batch_description.batch_features_types[name],
            )
            for name in self._batch_feature_names
        )
        self._batch_label_spec = tuple(
            tf.TensorSpec(
                shape=batch_description.batch_labels_shapes[name],
                dtype=batch_description.batch_labels_types[name],
            )
            for name in self._batch_label_names
        )

    @staticmethod
    def _pack(input: Dict[str, Any], names: Tuple[str, ...]) -> Tuple:
        return tuple(input.get(name) for name in names)

    def _pack_features(self, batch_features: Dict[str, Any]) -> Tuple:
        return self._pack(batch_features, self._batch_feature_names)

    def _pack_labels(self, batch_labels: Dict[str, Any]) -> Tuple:
        return self._pack(batch_labels, self._batch_label_names)

    @staticmethod
    def _unpack(input: Tuple, names: Tuple) -> Dict[str, Any]:
        return {name: value for name, value in zip(names, input) if value is not None}

    def _unpack_features(self, batch_features: Tuple) -> Dict[str, Any]:
        return self._unpack(batch_features, self._batch_feature_names)

    def _unpack_labels(self, batch_labels: Tuple) -> Dict[str, Any]:
        return self._unpack(batch_labels, self._batch_label_names)

    def build(self, input_shapes: Dict[str, Any]):
        graph_params = {
            name[4:]: value for name, value in self._params.items() if name.startswith("gnn_")
        }
        self._gnn = GNN(graph_params)
        self._gnn.build(
            GNNInput(
                node_features=self.get_initial_node_feature_shape(input_shapes),
                adjacency_lists=tuple(
                    input_shapes[f"adjacency_list_{edge_type_idx}"]
                    for edge_type_idx in range(self._num_edge_types)
                ),
                node_to_graph_map=tf.TensorShape((None,)),
                num_graphs=tf.TensorShape(()),
            )
        )

        super().build([])

        if not self._disable_tf_function_build:
            setattr(
                self,
                "_fast_run_step",
                tf.function(
                    input_signature=(
                        self._batch_feature_spec,
                        self._batch_label_spec,
                        tf.TensorSpec(shape=(), dtype=tf.bool),
                    )
                )(self._fast_run_step),
            )

    def get_initial_node_feature_shape(self, input_shapes) -> tf.TensorShape:
        return input_shapes["node_features"]

    def compute_initial_node_features(self, inputs, training: bool) -> tf.Tensor:
        return inputs["node_features"]

    @abstractmethod
    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]],
        training: bool,
    ) -> Any:
        """Compute task-specific output (labels, scores, regression values, ...).

        Args:
            batch_features: Input data for minibatch (as generated by the used datasets
                _finalise_batch method).
            final_node_representations:
                Per default (or if the hyperparameter "use_intermediate_gnn_results" was
                set to False), the final representations of the graph nodes as computed
                by the GNN.
                If the hyperparameter "use_intermediate_gnn_results" was set to True,
                a pair of the final node representation and all intermediate node
                representations, including the initial one.
            training: Flag indicating if we are training or not.

        Returns:
            Implementor's choice, but will be passed as task_output to compute_task_metrics
            during training/evaluation.
        """
        pass

    def compute_final_node_representations(self, inputs, training: bool):
        # Pack input data from keys back into a tuple:
        adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            inputs[f"adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )

        # Start the model computations:
        initial_node_features = self.compute_initial_node_features(inputs, training)
        gnn_input = GNNInput(
            node_features=initial_node_features,
            adjacency_lists=adjacency_lists,
            node_to_graph_map=inputs["node_to_graph_map"],
            num_graphs=inputs["num_graphs_in_batch"],
        )

        gnn_output = self._gnn(
            gnn_input,
            training=training,
            return_all_representations=self._use_intermediate_gnn_results
        )
        return gnn_output

    def call(self, inputs, training: bool):
        final_node_representations = self.compute_final_node_representations(inputs, training)
        return self.compute_task_output(inputs, final_node_representations, training)

    @abstractmethod
    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Compute task-specific loss & metrics (accuracy, F1 score, ...)

        Args:
            batch_features: Input data for minibatch (as generated by the used datasets
                _finalise_batch method).
            task_output: Output generated by compute_task_output.
            batch_labels: Target labels for minibatch (as generated by the used datasets
                _finalise_batch method).

        Returns:
            Dictionary of different metrics. Has to contain value for key
            "loss" (which will be used during training as starting point for backprop).
        """
        pass

    @abstractmethod
    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        """Compute single value used to measure quality of model at one epoch, where
        lower is better.
        This value, computed on the validation set, is used to determine if model
        training is still improving results.

        Args:
            task_results: List of results obtained by compute_task_metrics for the
                batches in one epoch.

        Returns:
            Pair of a metric value (lower ~ better) and a human-readable string
            describing it.
        """
        pass

    def _make_optimizer(
        self,
        learning_rate: Optional[
            Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]
        ] = None,
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create fresh optimizer.

        Args:
            learning_rate: Optional setting for learning rate; if unset, will
                use value from self._params["learning_rate"].
        """
        if learning_rate is None:
            learning_rate = self._params["learning_rate"]

            num_warmup_steps = self._params.get("learning_rate_warmup_steps")
            num_decay_steps = self._params.get("learning_rate_decay_steps")
            if num_warmup_steps is not None or num_decay_steps is not None:
                initial_learning_rate = 0.00001
                final_learning_rate = 0.00001
                if num_warmup_steps is None:
                    num_warmup_steps = -1  # Make sure that we have no warmup phase
                    initial_learning_rate = learning_rate
                if num_decay_steps is None:
                    num_decay_steps = (
                        1  # Value doesn't matter, but needs to be non-zero
                    )
                    final_learning_rate = learning_rate
                learning_rate = PolynomialWarmupAndDecaySchedule(
                    learning_rate=learning_rate,
                    warmup_steps=num_warmup_steps,
                    decay_steps=num_decay_steps,
                    initial_learning_rate=initial_learning_rate,
                    final_learning_rate=final_learning_rate,
                    power=1.0,
                )

        optimizer_name = self._params["optimizer"].lower()
        if optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=self._params["momentum"],
            )
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                momentum=self._params["momentum"],
                rho=self._params["rmsprop_rho"],
            )
        elif optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate,)
        else:
            raise Exception('Unknown optimizer "%s".' % (self._params["optimizer"]))

    def _apply_gradients(
        self, gradient_variable_pairs: Iterable[Tuple[tf.Tensor, tf.Variable]]
    ) -> None:
        """
        Apply gradients to the models variables during training.

        Args:
            gradient_variable_pairs: Iterable of pairs of gradients for a variable
                and the variable itself. Suitable to be fed into
                tf.keras.optimizer.*.apply_gradients.
        """
        if getattr(self, "_optimizer", None) is None:
            self._optimizer = self._make_optimizer()

        # Filter out variables without gradients:
        gradient_variable_pairs = [
            (grad, var) for (grad, var) in gradient_variable_pairs if grad is not None
        ]
        clip_val = self._params.get("gradient_clip_value")
        clip_norm_val = self._params.get("gradient_clip_norm")
        clip_global_norm_val = self._params.get("gradient_clip_global_norm")

        if clip_val is not None:
            if clip_norm_val is not None:
                raise ValueError("Both 'gradient_clip_value' and 'gradient_clip_norm' are set, but can only use one at a time.")
            if clip_global_norm_val is not None:
                raise ValueError("Both 'gradient_clip_value' and 'gradient_clip_global_norm' are set, but can only use one at a time.")
            gradient_variable_pairs = [
                (tf.clip_by_value(grad, -clip_val, clip_val), var)
                for (grad, var) in gradient_variable_pairs
            ]
        elif clip_norm_val is not None:
            if clip_global_norm_val is not None:
                raise ValueError("Both 'gradient_clip_norm' and 'gradient_clip_global_norm' are set, but can only use one at a time.")
            gradient_variable_pairs = [
                (tf.clip_by_norm(grad, clip_norm_val), var)
                for (grad, var) in gradient_variable_pairs
            ]
        elif clip_global_norm_val is not None:
            grads = [grad for (grad, _) in gradient_variable_pairs]
            clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_global_norm_val)
            gradient_variable_pairs = [
                (clipped_grad, var)
                for (clipped_grad, (_, var)) in zip(clipped_grads, gradient_variable_pairs)
            ]

        self._optimizer.apply_gradients(gradient_variable_pairs)

    # ----------------------------- Training Loop
    def _run_step(
        self,
        batch_features: Dict[str, tf.Tensor],
        batch_labels: Dict[str, tf.Tensor],
        training: bool,
    ) -> Dict[str, tf.Tensor]:
        batch_features_tuple = self._pack_features(batch_features)
        batch_labels_tuple = self._pack_labels(batch_labels)

        return self._fast_run_step(batch_features_tuple, batch_labels_tuple, training)

    def _fast_run_step(
        self,
        batch_features_tuple: Tuple[tf.Tensor],
        batch_labels_tuple: Tuple[tf.Tensor],
        training: bool,
    ):
        batch_features = self._unpack_features(batch_features_tuple)
        batch_labels = self._unpack_labels(batch_labels_tuple)

        with tf.GradientTape() as tape:
            task_output = self(batch_features, training=training)
            task_metrics = self.compute_task_metrics(
                batch_features=batch_features,
                task_output=task_output,
                batch_labels=batch_labels,
            )

        def _training_update():
            gradients = tape.gradient(task_metrics["loss"], self.trainable_variables)
            self._apply_gradients(zip(gradients, self.trainable_variables))
            self._train_step_counter.assign_add(1)

        def _no_op():
            pass

        tf.cond(training, true_fn=_training_update, false_fn=_no_op)

        return task_metrics

    def run_one_epoch(
        self, dataset: tf.data.Dataset, quiet: bool = False, training: bool = True,
    ) -> Tuple[float, float, List[Any]]:
        epoch_time_start = time.time()
        total_num_graphs = 0
        task_results = []
        total_loss = tf.constant(0, dtype=tf.float32)
        for step, (batch_features, batch_labels) in enumerate(dataset):
            task_metrics = self._run_step(batch_features, batch_labels, training)
            # task_metrics["loss"] is batch average loss over graphs
            # (loss per graph from self.compute_task_metrics())
            total_loss += task_metrics["loss"] * tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
            total_num_graphs += batch_features["num_graphs_in_batch"]
            task_results.append(task_metrics)

            if not quiet:
                epoch_graph_average_loss = (
                    total_loss / float(total_num_graphs)
                ).numpy()
                batch_graph_average_loss = task_metrics["loss"] 
                steps_per_second = step / (time.time() - epoch_time_start)
                print(
                    f"   Step: {step:4d}"
                    f"  |  Epoch graph avg. loss = {epoch_graph_average_loss:.5f}"
                    f"  |  Batch graph avg. loss = {batch_graph_average_loss:.5f}"
                    f"  |  Steps per sec = {steps_per_second:.5f}",
                    end="\r"
                )
        if not quiet:
            print("\r\x1b[K", end="")
        total_time = time.time() - epoch_time_start
        return total_loss / float(total_num_graphs), float(total_num_graphs) / total_time, task_results

    # ----------------------------- Prediction Loop
    def predict(self, dataset: tf.data.Dataset):
        task_outputs = []
        for batch_features, _ in dataset:
            task_outputs.append(self(batch_features, training=False))

        # Note: This assumes that the task output is a tensor (true for classification, regression,
        #  etc.) but subclasses implementing more complicated outputs will need to override this.
        return tf.concat(task_outputs, axis=0)

    def evaluate_model(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate the model using metrics that make sense for its application area.

        Args:
            dataset: A dataset to evaluate on, same format as used in the training loop.

        Returns:
            Dictionary mapping metric names (e.g., "accuracy", "roc_auc") to their respective
            values.
        """
        raise NotImplementedError()