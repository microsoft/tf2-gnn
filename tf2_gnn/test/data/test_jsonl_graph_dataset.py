"""Tests for the JsonLGraphPropertyDataset class."""
import os

import pytest
from dpu_utils.utils import RichPath

from tf2_gnn.data import JsonLGraphPropertyDataset, DataFold


@pytest.fixture
def jsonl_dataset():
    dataset_params = JsonLGraphPropertyDataset.get_default_hyperparameters()
    dataset = JsonLGraphPropertyDataset(dataset_params)
    data_path = RichPath.create(os.path.join(os.path.dirname(__file__), "..", "test_datasets"))
    dataset.load_data(data_path, folds_to_load=[DataFold.TRAIN, DataFold.VALIDATION])

    return dataset


def test_num_edge_types(jsonl_dataset: JsonLGraphPropertyDataset):
    # We expect 3 tied fwd/bkwd edge typess + 1 self-loop type:
    assert jsonl_dataset.num_edge_types == 4


def test_node_feature_shape(jsonl_dataset: JsonLGraphPropertyDataset):
    # Fixed in the test dataset:
    assert jsonl_dataset.node_feature_shape == (35,)


def test_num_loaded_data_elements(jsonl_dataset: JsonLGraphPropertyDataset):
    # Fixed number of data elements in the dataset:
    assert len(list(jsonl_dataset._graph_iterator(DataFold.TRAIN))) == 10
    assert len(list(jsonl_dataset._graph_iterator(DataFold.VALIDATION))) == 10


def test_batching(jsonl_dataset: JsonLGraphPropertyDataset):
    tf_dataset = jsonl_dataset.get_tensorflow_dataset(DataFold.TRAIN, use_worker_threads=False)

    tf_dataset_itererator = iter(tf_dataset)

    # Test that first minibatch has the right contents:
    first_minibatch = next(tf_dataset_itererator)
    (batch_features, batch_labels) = first_minibatch
    assert len(batch_features.keys()) == 7
    assert "node_features" in batch_features
    assert "node_to_graph_map" in batch_features
    assert "num_graphs_in_batch" in batch_features
    for edge_type_idx in range(4):
        assert f"adjacency_list_{edge_type_idx}" in batch_features

    assert batch_features["num_graphs_in_batch"] == 10

    assert len(batch_labels.keys()) == 1
    assert "target_value" in batch_labels

    try:
        next(tf_dataset_itererator)
        assert False  # iterator should be empty here
    except StopIteration:
        pass  # This is what we expect: The iterator should be finished.
