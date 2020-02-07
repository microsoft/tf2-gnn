"""General dataset class for datasets stored as JSONLines files."""
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphBatchTFDataDescription, GraphDataset, GraphSample

logger = logging.getLogger(__name__)


class GraphWithLabelSample(GraphSample):
    """Data structure holding a single graph with a single numeric property."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: List[np.ndarray],
        target_value: float,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._target_value = target_value

    @property
    def target_value(self) -> float:
        """Target value of the regression task."""
        return self._target_value

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"Target_value:   {self._target_value}"
        )


class JsonLGraphDataset(GraphDataset[GraphWithLabelSample]):
    """
    General class representing pre-split datasets in JSONLines format.
    Concretely, this class expects the following:
    * In the data directory, files "train.jsonl.gz", "valid.jsonl.gz" and
      "test.jsonl.gz" are used to store the train/valid/test datasets.
    * Each of the files is gzipped text file in which each line is a valid
      JSON dictionary with the following keys:
      - "Property": A floating point value (has to be 0.0/1.0 for
            classification data).
      - "graph": A dictionary with keys "node_features" (list of numerical
            initial node labels) and "adjacency_lists" (list of list of
            directed edge pairs)
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "for_classification": False,
            "max_nodes_per_batch": 10000,
            "num_fwd_edge_types": 4,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": True,
        }

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self._params = params
        self._for_classification = params["for_classification"]
        self._num_fwd_edge_types = params["num_fwd_edge_types"]
        if params["tie_fwd_bkwd_edges"]:
            self._num_edge_types = self._num_fwd_edge_types
        else:
            self._num_edge_types = 2 * self._num_fwd_edge_types
        self._num_edge_types += int(params["add_self_loop_edges"])

        self._node_feature_shape = None
        self._loaded_data: Dict[DataFold, List[GraphWithLabelSample]] = {}

    @property
    def name(self) -> str:
        return "JsonLGraphDataset"

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @classmethod
    def default_data_directory(cls):
        curr_dir = Path(os.path.abspath(inspect.getsourcefile(lambda: 0)))
        data_directory = os.path.join(curr_dir.parent.parent, "data")
        return data_directory

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        """Load the data from disk."""
        if path is None:
            path = RichPath.create(self.default_data_directory())
        logger.info("Starting to load data.")

        # If we haven't defined what folds to load, load all:
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("train.jsonl.gz"))
            logger.debug("Done loading training data.")
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("valid.jsonl.gz"))
            logger.debug("Done loading validation data.")
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path.join("test.jsonl.gz"))
            logger.debug("Done loading test data.")

    def __load_data(self, data_file: RichPath) -> List[GraphWithLabelSample]:
        return [
            self._process_raw_datapoint(datapoint) for datapoint in data_file.read_by_file_suffix()
        ]

    def _process_raw_datapoint(self, datapoint: Dict[str, Any]) -> GraphWithLabelSample:
        target_value = datapoint["Property"]
        if self._for_classification and target_value not in {0.0, 1.0}:
            raise ValueError(
                f"Loading classification data, but datapoint has target value {target_value} (expected 0.0 or 1.0)."
            )

        node_features = datapoint["graph"]["node_features"]
        num_nodes = len(node_features)

        raw_adjacency_lists = datapoint["graph"]["adjacency_lists"]

        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists, num_nodes
        )

        return GraphWithLabelSample(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
            node_features=node_features,
            target_value=target_value,
        )

    def _process_raw_adjacency_lists(
        self, raw_adjacency_lists: List[List[Tuple]], num_nodes: int
    ) -> Tuple[List, np.ndarray]:

        type_to_adj_list = [
            [] for _ in range(self._num_fwd_edge_types + int(self.params["add_self_loop_edges"]))
        ]  # type: List[List[Tuple[int, int]]]
        type_to_num_incoming_edges = np.zeros(shape=(self.num_edge_types, num_nodes))
        for raw_edge_type, edges in enumerate(raw_adjacency_lists):
            if self.params["add_self_loop_edges"]:
                fwd_edge_type = raw_edge_type + 1  # 0 will be the self-loop type
            else:
                fwd_edge_type = raw_edge_type  # Make edges start from 0
            for src, dest in edges:
                type_to_adj_list[fwd_edge_type].append((src, dest))
                type_to_num_incoming_edges[fwd_edge_type, dest] += 1
                if self.params["tie_fwd_bkwd_edges"]:
                    type_to_adj_list[fwd_edge_type].append((dest, src))
                    type_to_num_incoming_edges[fwd_edge_type, src] += 1

        if self.params["add_self_loop_edges"]:
            # Add self-loop edges (idx 0, which isn't used in the data):
            for node in range(num_nodes):
                type_to_num_incoming_edges[0, node] = 1
                type_to_adj_list[0].append((node, node))

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params["tie_fwd_bkwd_edges"]):
            # for (edge_type, adj_list) in enumerate(type_to_adj_list):
            num_edge_types_in_adj_lists = len(type_to_adj_list)
            for edge_type in range(num_edge_types_in_adj_lists):
                adj_list = type_to_adj_list[edge_type]
                # Don't add self loops again!
                if edge_type == 0 and self.params["add_self_loop_edges"]:
                    continue
                bkwd_edge_type = len(type_to_adj_list)
                type_to_adj_list.append([(y, x) for (x, y) in adj_list])
                for (x, y) in adj_list:
                    type_to_num_incoming_edges[bkwd_edge_type][y] += 1

        # Convert the adjacency lists to numpy arrays.
        type_to_adj_list = [
            np.array(adj_list, dtype=np.int32)
            if len(adj_list) > 0
            else np.zeros(shape=(0, 2), dtype=np.int32)
            for adj_list in type_to_adj_list
        ]
        return type_to_adj_list, type_to_num_incoming_edges

    @property
    def node_feature_shape(self) -> Tuple:
        """Return the shape of the node features."""
        if self._node_feature_shape is None:
            some_data_fold = next(iter(self._loaded_data.values()))
            self._node_feature_shape = (len(some_data_fold[0].node_features[0]),)
        return self._node_feature_shape

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphWithLabelSample]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(self._loaded_data[data_fold])
        return iter(self._loaded_data[data_fold])

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["target_value"] = []
        return new_batch

    def _add_graph_to_batch(
        self, raw_batch: Dict[str, Any], graph_sample: GraphWithLabelSample
    ) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["target_value"].append(graph_sample.target_value)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        return batch_features, {"target_value": raw_batch["target_value"]}

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "target_value": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "target_value": (None,)},
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
