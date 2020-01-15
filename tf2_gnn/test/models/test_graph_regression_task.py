"""Tests for the GraphRegressionTask class."""
import os
import random

import pytest
import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from tf2_gnn.data import GraphDataset, JsonLGraphDataset, DataFold
from tf2_gnn.models import GraphRegressionTask


@pytest.fixture
def jsonl_dataset():
    dataset_params = JsonLGraphDataset.get_default_hyperparameters()
    dataset = JsonLGraphDataset(dataset_params)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), "..", "test_datasets")
    )
    dataset.load_data(data_path, folds_to_load=[DataFold.TRAIN, DataFold.VALIDATION])

    return dataset


def test_train_improvement(jsonl_dataset: GraphDataset):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Turn off warnings in TF model construction, which are expected noise:
    def ignore_warn(*args, **kwargs):
        pass
    import warnings
    original_warn, warnings.warn = warnings.warn, ignore_warn

    model = GraphRegressionTask(
        GraphRegressionTask.get_default_hyperparameters(),
        num_edge_types=jsonl_dataset.num_edge_types,
    )
    data_description = jsonl_dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    # We run once on validation, do one training epoch, and then assert that results have improved:
    valid0_loss, _, valid0_results = model.run_one_epoch(
        dataset=jsonl_dataset.get_tensorflow_dataset(DataFold.VALIDATION),
        training=False,
        quiet=True,
    )
    valid0_metric, _ = model.compute_epoch_metrics(valid0_results)

    train1_loss, _, train1_results = model.run_one_epoch(
        dataset=jsonl_dataset.get_tensorflow_dataset(DataFold.TRAIN),
        training=True,
        quiet=True,
    )
    train1_metric, _ = model.compute_epoch_metrics(train1_results)

    valid1_loss, _, valid1_results = model.run_one_epoch(
        dataset=jsonl_dataset.get_tensorflow_dataset(DataFold.VALIDATION),
        training=False,
        quiet=True,
    )
    valid1_metric, _ = model.compute_epoch_metrics(valid1_results)

    assert valid0_loss > valid1_loss
    assert valid0_metric > valid1_metric

    train2_loss, _, train2_results = model.run_one_epoch(
        dataset=jsonl_dataset.get_tensorflow_dataset(DataFold.TRAIN),
        training=True,
        quiet=True,
    )
    train2_metric, _ = model.compute_epoch_metrics(train2_results)

    assert train1_loss > train2_loss
    assert train1_metric > train2_metric
