"""Task for regression on QM9 dataset."""
from typing import Any, Dict, List, Tuple, Optional, Union

import tensorflow as tf
from dpu_utils.tf2utils import MLP

from tf2_gnn.data import GraphDataset, QM9Dataset
from tf2_gnn.models import GraphTaskModel


# These magic constants were obtained during dataset generation, as result of normalising
# the values of target properties:
CHEMICAL_ACC_NORMALISING_FACTORS = [
    0.066513725,
    0.012235489,
    0.071939046,
    0.033730778,
    0.033486113,
    0.004278493,
    0.001330901,
    0.004165489,
    0.004128926,
    0.00409976,
    0.004527465,
    0.012292586,
    0.037467458,
]


class QM9RegressionTask(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(
        cls, mp_style: Optional[str] = None
    ) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "use_intermediate_gnn_results": False,
            "out_layer_dropout_keep_prob": 1.0,
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None, **kwargs):
        super().__init__(params, dataset=dataset, name=name, **kwargs)
        assert isinstance(dataset, QM9Dataset)

        self._task_id = int(dataset._params["task_id"])

        self._regression_gate = MLP(
            out_size=1,
            hidden_layers=[],
            use_biases=True,
            dropout_rate=self._params["out_layer_dropout_keep_prob"],
            name="gate",
        )
        self._regression_transform = MLP(
            out_size=1,
            hidden_layers=[],
            use_biases=True,
            dropout_rate=self._params["out_layer_dropout_keep_prob"],
            name="transform"
        )

    def build(self, input_shapes):
        with tf.name_scope(self.__class__.__name__):
            with tf.name_scope("node_gate"):
                self._regression_gate.build(
                    tf.TensorShape(
                        (
                            None,
                            input_shapes["node_features"][-1]
                            + self._params["gnn_hidden_dim"],
                        )
                    )
                )
            with tf.name_scope("node_transform"):
                self._regression_transform.build(
                    tf.TensorShape((None, self._params["gnn_hidden_dim"]))
                )

        super().build(input_shapes)

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: Union[tf.Tensor, Tuple[tf.Tensor, List[tf.Tensor]]],
        training: bool,
    ) -> Any:
        if self._params["use_intermediate_gnn_results"]:
            final_node_representations, _ = final_node_representations

        # The per-node regression uses only final node representations:
        per_node_output = self._regression_transform(
            final_node_representations, training=training
        )  # Shape [V, 1]

        # The gating uses both initial and final node representations:
        per_node_weight = self._regression_gate(
            tf.concat(
                [batch_features["node_features"], final_node_representations], axis=-1
            ),
            training=training,
        )  # Shape [V, 1]

        per_node_weighted_output = tf.squeeze(
            tf.nn.sigmoid(per_node_weight) * per_node_output, axis=-1
        )  # Shape [V]
        per_graph_output = tf.math.unsorted_segment_sum(
            data=per_node_weighted_output,
            segment_ids=batch_features["node_to_graph_map"],
            num_segments=batch_features["num_graphs_in_batch"],
        )  # Shape [G]

        return per_graph_output

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
        return (
            epoch_mae,
            (
                f"Task {self._task_id} |"
                f" MSE = {epoch_mse:.3f} |"
                f" MAE = {epoch_mae:.3f} |"
                f" Error Ratio: {epoch_mae / CHEMICAL_ACC_NORMALISING_FACTORS[self._task_id]:.3f}"
            ),
        )
