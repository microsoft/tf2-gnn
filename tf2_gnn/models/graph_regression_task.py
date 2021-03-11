"""General task for GNN regression."""
from typing import Any, Dict, List, Tuple, Optional, Union

import tensorflow as tf
from dpu_utils.tf2utils import MLP

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel
from tf2_gnn.layers import (
    WeightedSumGraphRepresentation,
    NodesToGraphRepresentationInput,
)


class GraphRegressionTask(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(
        cls, mp_style: Optional[str] = None
    ) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "use_intermediate_gnn_results": True,
            "graph_aggregation_output_size": 32,
            "graph_aggregation_num_heads": 4,
            "graph_aggregation_layers": [32, 32],
            "graph_aggregation_dropout_rate": 0.1,
            "regression_mlp_layers": [64, 32],
            "regression_mlp_dropout": 0.1,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None, **kwargs):
        super().__init__(params, dataset=dataset, name=name, **kwargs)
        self._node_to_graph_aggregation = None

        # Construct sublayers:
        self._weighted_avg_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self._params["graph_aggregation_output_size"],
            num_heads=self._params["graph_aggregation_num_heads"],
            weighting_fun="softmax",
            scoring_mlp_layers=self._params["graph_aggregation_layers"],
            scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
            scoring_mlp_activation_fun="elu",
            transformation_mlp_layers=self._params["graph_aggregation_layers"],
            transformation_mlp_dropout_rate=self._params[
                "graph_aggregation_dropout_rate"
            ],
            transformation_mlp_activation_fun="elu",
        )
        self._weighted_sum_of_nodes_to_graph_repr = WeightedSumGraphRepresentation(
            graph_representation_size=self._params["graph_aggregation_output_size"],
            num_heads=self._params["graph_aggregation_num_heads"],
            weighting_fun="sigmoid",
            scoring_mlp_layers=self._params["graph_aggregation_layers"],
            scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
            scoring_mlp_activation_fun="elu",
            transformation_mlp_layers=self._params["graph_aggregation_layers"],
            transformation_mlp_dropout_rate=self._params[
                "graph_aggregation_dropout_rate"
            ],
            transformation_mlp_activation_fun="elu",
        )

        self._regression_mlp = MLP(
            out_size=1,
            hidden_layers=self._params["regression_mlp_layers"],
            dropout_rate=self._params["regression_mlp_dropout"],
            use_biases=True,
            activation_fun=tf.nn.relu,
        )

    def build(self, input_shapes):
        if self._params["use_intermediate_gnn_results"]:
            # We get the initial GNN input + results for all layers:
            node_repr_size = (
                input_shapes["node_features"][-1]
                + self._params["gnn_hidden_dim"] * self._params["gnn_num_layers"]
            )
        else:
            node_repr_size = (
                input_shapes["node_features"][-1] + self._params["gnn_hidden_dim"]
            )

        node_to_graph_repr_input = NodesToGraphRepresentationInput(
            node_embeddings=tf.TensorShape((None, node_repr_size)),
            node_to_graph_map=tf.TensorShape((None)),
            num_graphs=tf.TensorShape(()),
        )

        with tf.name_scope(self.__class__.__name__):
            with tf.name_scope("graph_representation_computation"):
                with tf.name_scope("weighted_avg"):
                    self._weighted_avg_of_nodes_to_graph_repr.build(
                        node_to_graph_repr_input
                    )
                with tf.name_scope("weighted_sum"):
                    self._weighted_sum_of_nodes_to_graph_repr.build(
                        node_to_graph_repr_input
                    )

            self._regression_mlp.build(
                tf.TensorShape(
                    (None, 2 * self._params["graph_aggregation_output_size"])
                )
            )

        super().build(input_shapes)

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> Any:
        if self._params["use_intermediate_gnn_results"]:
            _, intermediate_node_representations = final_node_representations
            # We want to skip the first "intermediate" representation, which is the output of
            # the initial feature -> GNN input layer:
            node_representations = tf.concat(
                (batch_features["node_features"],)
                + intermediate_node_representations[1:],
                axis=-1,
            )
        else:
            node_representations = tf.concat(
                [batch_features["node_features"], final_node_representations], axis=-1
            )

        graph_representation_layer_input = NodesToGraphRepresentationInput(
            node_embeddings=node_representations,
            node_to_graph_map=batch_features["node_to_graph_map"],
            num_graphs=batch_features["num_graphs_in_batch"],
        )
        weighted_avg_graph_repr = self._weighted_avg_of_nodes_to_graph_repr(
            graph_representation_layer_input, training=training
        )
        weighted_sum_graph_repr = self._weighted_sum_of_nodes_to_graph_repr(
            graph_representation_layer_input, training=training
        )

        graph_representations = tf.concat(
            [weighted_avg_graph_repr, weighted_sum_graph_repr], axis=-1
        )  # shape: [G, GD]

        per_graph_results = self._regression_mlp(
            graph_representations, training=training
        )  # shape: [G, 1]

        return tf.squeeze(per_graph_results, axis=-1)

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        mse = tf.losses.mean_squared_error(batch_labels["target_value"], task_output)
        mae = tf.losses.mean_absolute_error(batch_labels["target_value"], task_output)
        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        return {
            "loss": mse,
            "batch_squared_error": mse * num_graphs,
            "batch_absolute_error": mae * num_graphs,
            "num_graphs": num_graphs,
        }

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_graphs = sum(
            batch_task_result["num_graphs"] for batch_task_result in task_results
        )
        total_absolute_error = sum(
            batch_task_result["batch_absolute_error"]
            for batch_task_result in task_results
        )
        total_squared_error = sum(
            batch_task_result["batch_squared_error"]
            for batch_task_result in task_results
        )
        epoch_mse = (total_squared_error / total_num_graphs).numpy()
        epoch_mae = (total_absolute_error / total_num_graphs).numpy()
        return epoch_mae, f" MSE = {epoch_mse:.3f} | MAE = {epoch_mae:.3f}"

    def evaluate_model(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        import sklearn.metrics as metrics

        predictions = self.predict(dataset).numpy()
        labels = []
        for _, batch_labels in dataset:
            labels.append(batch_labels["target_value"])
        labels = tf.concat(labels, axis=0).numpy()

        metrics = dict(
            mae=metrics.mean_absolute_error(y_true=labels, y_pred=predictions),
            mse=metrics.mean_squared_error(y_true=labels, y_pred=predictions),
            max_err=metrics.max_error(y_true=labels, y_pred=predictions),
            expl_var=metrics.explained_variance_score(
                y_true=labels, y_pred=predictions
            ),
            r2_score=metrics.r2_score(y_true=labels, y_pred=predictions),
        )

        return metrics
