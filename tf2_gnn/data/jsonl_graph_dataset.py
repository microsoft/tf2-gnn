"""General dataset class for datasets stored as JSONLines files."""
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set

import numpy as np
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphDataset, GraphSampleType, GraphSample
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists

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
        self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ):
        super().__init__(params, metadata=metadata, **kwargs)
        self._params = params
        self._num_fwd_edge_types = params["num_fwd_edge_types"]

        self._tied_fwd_bkwd_edge_types = get_tied_edge_types(
            tie_fwd_bkwd_edges=params["tie_fwd_bkwd_edges"],
            num_fwd_edge_types=params["num_fwd_edge_types"],
        )

        self._num_edge_types = compute_number_of_edge_types(
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            num_fwd_edge_types=self._num_fwd_edge_types,
            add_self_loop_edges=params["add_self_loop_edges"],
        )

        self._loaded_data: Dict[DataFold, List[GraphSampleType]] = {}

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def node_feature_shape(self) -> Tuple:
        """Return the shape of the node features."""
        node_feature_shape = self.metadata.get("_node_feature_shape")
        if node_feature_shape is None:
            some_data_fold = next(iter(self._loaded_data.values()))
            node_feature_shape = (len(some_data_fold[0].node_features[0]),)
            self.metadata["_node_feature_shape"] = node_feature_shape
        return node_feature_shape

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
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        return process_adjacency_lists(
            adjacency_lists=raw_adjacency_lists,
            num_nodes=num_nodes,
            add_self_loop_edges=self.params["add_self_loop_edges"],
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
        )

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphSampleType]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(self._loaded_data[data_fold])
        return iter(self._loaded_data[data_fold])
