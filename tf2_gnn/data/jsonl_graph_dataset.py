"""General dataset class for datasets stored as JSONLines files."""
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set

import numpy as np
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphDataset, GraphSampleType, GraphSample

logger = logging.getLogger(__name__)


class JsonLGraphDataset(GraphDataset[GraphSampleType]):
    """
    General class representing pre-split datasets in JSONLines format.
    Concretely, this class expects the following:
    * In the data directory, files "train.jsonl.gz", "valid.jsonl.gz" and
      "test.jsonl.gz" are used to store the train/valid/test datasets.
    * Each of the files is gzipped text file in which each line is a valid
      JSON dictionary with a "graph" key, which in turn points to a
      dictionary with keys
       - "node_features" (list of numerical initial node labels)
       - "adjacency_lists" (list of list of directed edge pairs)
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            "num_fwd_edge_types": 3,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": True,
        }
        super_hypers.update(this_hypers)
        return super_hypers

    def __init__(
        self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(params, metadata=metadata)
        self._params = params
        self._num_fwd_edge_types = params["num_fwd_edge_types"]
        if params["tie_fwd_bkwd_edges"]:
            self._num_edge_types = self._num_fwd_edge_types
        else:
            self._num_edge_types = 2 * self._num_fwd_edge_types
        self._num_edge_types += int(params["add_self_loop_edges"])

        self._node_feature_shape: Optional[Tuple[int]] = None
        self._loaded_data: Dict[DataFold, List[GraphSampleType]] = {}

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def node_feature_shape(self) -> Tuple:
        """Return the shape of the node features."""
        if self._node_feature_shape is None:
            some_data_fold = next(iter(self._loaded_data.values()))
            self._node_feature_shape = (len(some_data_fold[0].node_features[0]),)
        return self._node_feature_shape

    def load_metadata(self, path: RichPath) -> None:
        """Load the metadata for a dataset (such as vocabularies, names of properties, ...)
        from a path on disk.

        Note: Implementors needing to act on metadata information before loading any actual data
        should override this method.
        """
        if self.metadata == {}:
            metadata_path = path.join("metadata.pkl.gz")
            if metadata_path.exists():
                logger.info(f"Loading metadata from {metadata_path}")
                self._metadata = metadata_path.read_by_file_suffix()
        else:
            logger.warning("Using metadata passed to constructor, not metadata stored with data.")

    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        """Load the data from disk."""
        logger.info(f"Starting to load data from {path}.")
        self.load_metadata(path)

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

    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        if target_fold not in self._loaded_data:
            self._loaded_data[target_fold] = []
        for datapoint in datapoints:
            self._loaded_data[target_fold].append(self._process_raw_datapoint(datapoint))

    def __load_data(self, data_file: RichPath) -> List[GraphSampleType]:
        return [
            self._process_raw_datapoint(datapoint) for datapoint in data_file.read_by_file_suffix()
        ]

    def _process_raw_datapoint(self, datapoint: Dict[str, Any]) -> GraphSampleType:
        node_features = datapoint["graph"]["node_features"]
        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists=datapoint["graph"]["adjacency_lists"],
            num_nodes=len(node_features),
        )

        return GraphSample(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_inedges=type_to_num_incoming_edges,
            node_features=node_features,
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

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphSampleType]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(self._loaded_data[data_fold])
        return iter(self._loaded_data[data_fold])
