"""General task for graph binary classification."""
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models.graph_regression_task import GraphRegressionTask


class GraphBinaryClassificationTask(GraphRegressionTask):
    @classmethod
    def get_default_hyperparameters(
        cls, mp_style: Optional[str] = None
    ) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {}
        super_params.update(these_hypers)
        return super_params

    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: tf.Tensor,
        training: bool,
    ) -> Any:
        per_graph_regression_results = super().compute_task_output(
            batch_features, final_node_representations, training
        )

        return tf.nn.sigmoid(per_graph_regression_results)

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        ce = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true=batch_labels["target_value"],
                y_pred=task_output,
                from_logits=False,
            )
        )
        num_correct = tf.reduce_sum(
            tf.cast(
                tf.math.equal(batch_labels["target_value"], tf.math.round(task_output)),
                tf.int32,
            )
        )
        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        return {
            "loss": ce,
            "batch_acc": tf.cast(num_correct, tf.float32) / num_graphs,
            "num_correct": num_correct,
            "num_graphs": num_graphs,
        }

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_graphs = np.sum(
            batch_task_result["num_graphs"] for batch_task_result in task_results
        )
        total_num_correct = np.sum(
            batch_task_result["num_correct"] for batch_task_result in task_results
        )
        epoch_acc = tf.cast(total_num_correct, tf.float32) / total_num_graphs
        return -epoch_acc.numpy(), f"Accuracy = {epoch_acc.numpy():.3f}"

    def evaluate_model(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        import sklearn.metrics as metrics

        predictions = self.predict(dataset).numpy()
        rounded_preds = np.round(predictions)
        labels = []
        for _, batch_labels in dataset:
            labels.append(batch_labels["target_value"])
        labels = tf.concat(labels, axis=0).numpy()

        try:
            roc_auc = metrics.roc_auc_score(y_true=labels, y_score=predictions)
            average_precision = metrics.average_precision_score(
                y_true=labels, y_score=predictions
            )
        except:
            roc_auc = np.nan
            average_precision = np.nan

        metrics = dict(
            acc=metrics.accuracy_score(y_true=labels, y_pred=rounded_preds),
            balanced_acc=metrics.balanced_accuracy_score(
                y_true=labels, y_pred=rounded_preds
            ),
            precision=metrics.precision_score(y_true=labels, y_pred=rounded_preds),
            recall=metrics.recall_score(y_true=labels, y_pred=rounded_preds),
            f1_score=metrics.f1_score(y_true=labels, y_pred=rounded_preds),
            roc_auc=roc_auc,
            average_precision=average_precision,
        )

        return metrics
