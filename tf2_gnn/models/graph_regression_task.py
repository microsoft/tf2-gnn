"""General task for GNN regression."""
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.tf2utils import MLP

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel
from tf2_gnn.layers import WeightedSumGraphRepresentation, NodesToGraphRepresentationInput


class GraphRegressionTask(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "graph_aggregation_num_heads": 4,
            "graph_aggregation_hidden_layers": [128],
            "graph_aggregation_dropout_rate": 0.2,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
        self._node_to_graph_aggregation = None

    def build(self, input_shapes):
        with tf.name_scope(self._name):
            self._node_to_graph_repr_layer = WeightedSumGraphRepresentation(
                graph_representation_size=self._params["graph_aggregation_num_heads"],
                num_heads=self._params["graph_aggregation_num_heads"],
                scoring_mlp_layers=self._params["graph_aggregation_hidden_layers"],
                scoring_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
                transformation_mlp_layers=self._params["graph_aggregation_hidden_layers"],
                transformation_mlp_dropout_rate=self._params["graph_aggregation_dropout_rate"],
            )
            self._node_to_graph_repr_layer.build(
                NodesToGraphRepresentationInput(
                    node_embeddings=tf.TensorShape(
                        (None, input_shapes["node_features"][-1] + self._params["gnn_hidden_dim"])
                    ),
                    node_to_graph_map=tf.TensorShape((None)),
                    num_graphs=tf.TensorShape(()),
                )
            )

        super().build(input_shapes)

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: tf.Tensor,
        training: bool,
    ) -> Any:
        per_graph_results = self._node_to_graph_repr_layer(
            NodesToGraphRepresentationInput(
                node_embeddings=tf.concat(
                    [batch_features["node_features"], final_node_representations], axis=-1
                ),
                node_to_graph_map=batch_features["node_to_graph_map"],
                num_graphs=batch_features["num_graphs_in_batch"],
            )
        )  # Shape [G, graph_aggregation_num_heads]
        per_graph_results = tf.reduce_sum(per_graph_results, axis=-1)  # Shape [G]

        return per_graph_results

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
            batch_task_result["batch_absolute_error"] for batch_task_result in task_results
        )
        epoch_mae = total_absolute_error / total_num_graphs
        return epoch_mae.numpy(), f"Mean Absolute Error = {epoch_mae.numpy():.3f}"
