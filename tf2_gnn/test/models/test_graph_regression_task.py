"""Tests for the GraphRegressionTask class."""
import os
import random
import warnings
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from dpu_utils.utils import RichPath

from tf2_gnn.data import DataFold, GraphDataset, JsonLGraphPropertyDataset
from tf2_gnn.models import GraphRegressionTask


# Turn off warnings in TF model construction, which are expected noise:
def ignore_warn(*args, **kwargs):
    pass


original_warn, warnings.warn = warnings.warn, ignore_warn


@pytest.fixture
def jsonl_dataset():
    dataset_params = JsonLGraphPropertyDataset.get_default_hyperparameters()
    dataset = JsonLGraphPropertyDataset(dataset_params)
    data_path = RichPath.create(
        os.path.join(os.path.dirname(__file__), "..", "test_datasets")
    )
    dataset.load_data(data_path, folds_to_load=[DataFold.TRAIN, DataFold.VALIDATION])

    return dataset


@pytest.fixture
def weights_file():
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
    os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "weights.h5")
    yield save_path
    # Tear down:
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(save_dir):
        os.rmdir(save_dir)


def test_weights_save_without_error(jsonl_dataset: GraphDataset, weights_file: str):
    model = GraphRegressionTask(
        GraphRegressionTask.get_default_hyperparameters(),
        dataset=jsonl_dataset,
    )
    data_description = jsonl_dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    model.save_weights(weights_file, save_format="h5")


def test_weights_load_from_file(jsonl_dataset: GraphDataset, weights_file: str):
    # Clear the Keras session so that unique naming does not mess up weight loading.
    tf.keras.backend.clear_session()

    # Build a model and save its (random) weights.
    data_description = jsonl_dataset.get_batch_tf_data_description()
    model_1 = GraphRegressionTask(
        GraphRegressionTask.get_default_hyperparameters(),
        dataset=jsonl_dataset,
    )
    model_1.build(data_description.batch_features_shapes)
    model_1.save_weights(weights_file, save_format="h5")

    weights_1: List[tf.Variable] = [x.numpy() for x in model_1.trainable_variables]

    # Clear the Keras session so that unique naming does not mess up weight loading.
    del model_1
    tf.keras.backend.clear_session()

    # Build a second model and load the first set of weights into it.
    model_2 = GraphRegressionTask(
        GraphRegressionTask.get_default_hyperparameters(),
        dataset=jsonl_dataset,
    )
    model_2.build(data_description.batch_features_shapes)

    model_2.load_weights(weights_file, by_name=True)
    weights_2: List[np.ndarray] = [x.numpy() for x in model_2.trainable_variables]

    for w_1, w_2 in zip(weights_1, weights_2):
        np.testing.assert_array_equal(w_1, w_2)


def test_train_improvement(jsonl_dataset: GraphDataset):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    model = GraphRegressionTask(
        GraphRegressionTask.get_default_hyperparameters(),
        dataset=jsonl_dataset,
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
