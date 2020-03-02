import time
from abc import abstractmethod
from typing import Tuple, List, Dict, Optional, Any, Iterable, Union

import tensorflow as tf

from tf2_gnn import GNNInput, GNN
from tf2_gnn.data import GraphDataset


class GraphTaskModel(tf.keras.Model):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        """Get the default hyperparameter dictionary for the class."""
        params = {f"gnn_{name}": value for name, value in GNN.get_default_hyperparameters(mp_style).items()}
        these_hypers: Dict[str, Any] = {
            "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
            "learning_rate": 0.001,
            "learning_rate_decay": 0.98,
            "momentum": 0.85,
            "gradient_clip_value": 1.0,
            "use_intermediate_gnn_results": False,
        }
        params.update(these_hypers)
        return params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(name=name)
        self._params = params
        self._num_edge_types = dataset.num_edge_types
        self._use_intermediate_gnn_results = params.get("use_intermediate_gnn_results", False)
        self._train_step_counter = 0

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

    def get_initial_node_feature_shape(self, input_shapes) -> tf.TensorShape:
        return input_shapes["node_features"]

    def compute_initial_node_features(self, inputs, training: bool) -> tf.Tensor:
        return inputs["node_features"]

    @abstractmethod
    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
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

    def call(self, inputs, training: bool):
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
        return self.compute_task_output(inputs, gnn_output, training)

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
            task_output: Output geerated by compute_task_output.
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

        optimizer_name = self._params["optimizer"].lower()
        if optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=self._params["momentum"],
                clipvalue=self._params["gradient_clip_value"],
            )
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                decay=self._params["learning_rate_decay"],
                momentum=self._params["momentum"],
                clipvalue=self._params["gradient_clip_value"],
            )
        elif optimizer_name == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipvalue=self._params["gradient_clip_value"],
            )
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

        self._optimizer.apply_gradients(gradient_variable_pairs)

    # ----------------------------- Training Loop
    def run_one_epoch(
        self, dataset: tf.data.Dataset, quiet: bool = False, training: bool = True,
    ) -> Tuple[float, float, List[Any]]:
        epoch_time_start = time.time()
        total_num_graphs = 0
        task_results = []
        total_loss = tf.constant(0, dtype=tf.float32)
        for step, (batch_features, batch_labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                task_output = self(batch_features, training=training)
                task_metrics = self.compute_task_metrics(batch_features, task_output, batch_labels)
            total_loss += task_metrics["loss"]
            total_num_graphs += batch_features["num_graphs_in_batch"]
            task_results.append(task_metrics)

            if training:
                gradients = tape.gradient(
                    task_metrics["loss"], self.trainable_variables
                )
                self._apply_gradients(zip(gradients, self.trainable_variables))
                self._train_step_counter += 1

            if not quiet:
                epoch_graph_average_loss = (total_loss / float(total_num_graphs)).numpy()
                batch_graph_average_loss = task_metrics["loss"] / float(batch_features["num_graphs_in_batch"])
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
