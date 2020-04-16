"""Tests for the dataset classes."""
import json
import os
from typing import Any, List, NamedTuple, Tuple

import numpy as np
import pytest

from dpu_utils.utils import RichPath
from tf2_gnn.data.graph_dataset import DataFold, GraphDataset, GraphSampleType
from tf2_gnn.data.jsonl_graph_property_dataset import JsonLGraphPropertyDataset
from tf2_gnn.data.ppi_dataset import PPIDataset
from tf2_gnn.data.qm9_dataset import QM9Dataset


class TestExpectedValues(NamedTuple):
    num_edge_types: int
    node_feature_shape: Tuple[int]
    num_train_samples: int
    num_valid_samples: int
    labels_key_name: str
    add_self_loop_edges: bool
    tie_fwd_bkwd_edges: bool
    self_loop_edge_type: int


class TestCase(NamedTuple):
    dataset: GraphDataset[Any]
    expected: TestExpectedValues


@pytest.fixture
def tmp_data_dir():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
    os.mkdir(data_dir)

    yield data_dir

    os.rmdir(data_dir)


@pytest.fixture
def qm9_train_valid_paths(tmp_data_dir):
    train_valid_paths = [
        os.path.join(tmp_data_dir, f"{split}.jsonl.gz") for split in ["train", "valid"]
    ]

    data_samples = 5 * [
        {
            "graph": [(0, 1, 1)],  # Edge between vertices 0 and 1, with type 1.
            "node_features": [[1, 0], [0, 1]],  # Two nodes, with features of shape (2,).
            "targets": [[1.0]],  # Target value for the graph.
        }
    ]

    for path in train_valid_paths:
        RichPath.create(path).save_as_compressed_file(data_samples)

    yield train_valid_paths

    for path in train_valid_paths:
        os.remove(path)


@pytest.fixture
def ppi_train_valid_paths(tmp_data_dir):
    train_valid_paths = [
        {
            key: os.path.join(tmp_data_dir, f"{split}_{key}.{ext}")
            for (key, ext) in [
                ("graph", "json"),
                ("feats", "npy"),
                ("labels", "npy"),
                ("graph_id", "npy"),
            ]
        }
        for split in ["train", "valid"]
    ]

    for paths in train_valid_paths:
        with open(paths["graph"], "w") as f:
            # Edge between vertices 0 and 1.
            json.dump({"links": [{"source": 0, "target": 1}]}, f)

        # Two nodes, with features of shape (3,).
        np.save(paths["feats"], np.zeros((2, 3)))
        np.save(paths["labels"], np.zeros((2, 1)))

        # Both nodes are part of a single graph.
        np.save(paths["graph_id"], np.zeros((2,)))

    yield train_valid_paths

    for path in sum([list(p.values()) for p in train_valid_paths], []):
        os.remove(path)


@pytest.fixture
def jsonl_test_case():
    dataset_params = JsonLGraphPropertyDataset.get_default_hyperparameters()
    dataset = JsonLGraphPropertyDataset(dataset_params)
    data_path = RichPath.create(os.path.join(os.path.dirname(__file__), "..", "test_datasets"))
    dataset.load_data(data_path, folds_to_load={DataFold.TRAIN, DataFold.VALIDATION})

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=4,
            node_feature_shape=(35,),
            num_train_samples=10,
            num_valid_samples=10,
            labels_key_name="target_value",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=0,
        ),
    )


@pytest.fixture
def qm9_test_case(tmp_data_dir, qm9_train_valid_paths):
    dataset_params = QM9Dataset.get_default_hyperparameters()
    dataset = QM9Dataset(dataset_params)

    dataset.load_data(
        RichPath.create(tmp_data_dir), folds_to_load={DataFold.TRAIN, DataFold.VALIDATION}
    )

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=5,
            node_feature_shape=(2,),
            num_train_samples=5,
            num_valid_samples=5,
            labels_key_name="target_value",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=0,
        ),
    )


@pytest.fixture
def ppi_test_case(tmp_data_dir, ppi_train_valid_paths):
    dataset_params = PPIDataset.get_default_hyperparameters()
    dataset = PPIDataset(dataset_params)

    dataset.load_data(
        RichPath.create(tmp_data_dir), folds_to_load={DataFold.TRAIN, DataFold.VALIDATION}
    )

    return TestCase(
        dataset=dataset,
        expected=TestExpectedValues(
            num_edge_types=3,
            node_feature_shape=(3,),
            num_train_samples=1,
            num_valid_samples=1,
            labels_key_name="node_labels",
            add_self_loop_edges=dataset_params["add_self_loop_edges"],
            tie_fwd_bkwd_edges=dataset_params["tie_fwd_bkwd_edges"],
            self_loop_edge_type=1,
        ),
    )


# `pytest.mark.parametrize` only accepts a list of test samples as input, and not list of fixtures.
# This is a workaround which allows to get parametrization with fixtures.
@pytest.fixture(params=["jsonl_dataset", "qm9_dataset", "ppi_dataset"])
def test_case(request, jsonl_test_case, qm9_test_case, ppi_test_case):
    return {
        "jsonl_dataset": jsonl_test_case,
        "qm9_dataset": qm9_test_case,
        "ppi_dataset": ppi_test_case,
    }[request.param]


def test_num_edge_types(test_case: TestCase):
    assert test_case.dataset.num_edge_types == test_case.expected.num_edge_types


def test_node_feature_shape(test_case: TestCase):
    assert test_case.dataset.node_feature_shape == test_case.expected.node_feature_shape


def test_num_loaded_data_elements(test_case: TestCase):
    train_data = list(test_case.dataset._graph_iterator(DataFold.TRAIN))
    valid_data = list(test_case.dataset._graph_iterator(DataFold.VALIDATION))

    assert len(train_data) == test_case.expected.num_train_samples
    assert len(valid_data) == test_case.expected.num_valid_samples


def test_batching(test_case: TestCase):
    tf_dataset = test_case.dataset.get_tensorflow_dataset(DataFold.TRAIN, use_worker_threads=False)

    tf_dataset_itererator = iter(tf_dataset)

    # Test that first minibatch has the right contents:
    first_minibatch = next(tf_dataset_itererator)
    (batch_features, batch_labels) = first_minibatch

    assert len(batch_features.keys()) == 3 + test_case.expected.num_edge_types

    assert "node_features" in batch_features
    assert "node_to_graph_map" in batch_features
    assert "num_graphs_in_batch" in batch_features

    for edge_type_idx in range(test_case.expected.num_edge_types):
        assert f"adjacency_list_{edge_type_idx}" in batch_features

    assert batch_features["num_graphs_in_batch"] == test_case.expected.num_train_samples

    assert len(batch_labels.keys()) == 1
    assert test_case.expected.labels_key_name in batch_labels

    try:
        next(tf_dataset_itererator)
        assert False  # iterator should be empty here
    except StopIteration:
        pass  # This is what we expect: The iterator should be finished.


def get_sorted_lists_of_edges(graph_sample: GraphSampleType) -> List[List[Tuple[int, int]]]:
    return [sorted(tuple(edge) for edge in adj) for adj in graph_sample.adjacency_lists]


def test_added_self_loop_edges(test_case: TestCase):
    for datapoint in test_case.dataset._graph_iterator(DataFold.TRAIN):
        adjacency_lists = get_sorted_lists_of_edges(datapoint)

        for (edge_type, adjacency_list) in enumerate(adjacency_lists):
            if (
                test_case.expected.add_self_loop_edges
                and edge_type == test_case.expected.self_loop_edge_type
            ):
                num_nodes = len(datapoint.node_features)
                assert adjacency_list == [(i, i) for i in range(num_nodes)]
            else:
                for (src, dest) in adjacency_list:
                    # If self loops were not explicitly added, expect no self loops in the graph.
                    # This assumption may not universally hold, but it does for the datasets tested
                    # here.
                    assert src != dest


def test_tied_fwd_bkwd_edges(test_case: TestCase):
    for datapoint in test_case.dataset._graph_iterator(DataFold.TRAIN):
        adjacency_lists = get_sorted_lists_of_edges(datapoint)

        for adjacency_list in adjacency_lists:
            adjacency_list_flipped = sorted([(dest, src) for (src, dest) in adjacency_list])

            # This will hold even if `adjacency_list` corresponds to self-loops.
            assert adjacency_list_flipped in adjacency_lists
