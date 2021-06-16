from typing import Any, Dict, Iterable, List, Iterator, Tuple, Optional, Set

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists


class PPIGraphSample(GraphSample):
    """Data structure holding a single PPI graph."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_inedges: np.ndarray,
        node_features: np.ndarray,
        node_labels: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_inedges, node_features)
        self._node_labels = node_labels

    @property
    def node_labels(self) -> np.ndarray:
        """Node labels to predict as ndarray of shape [V, C]"""
        return self._node_labels


class PPIDataset(GraphDataset[PPIGraphSample]):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            "max_nodes_per_batch": 10000,
            "add_self_loop_edges": True,
            "tie_fwd_bkwd_edges": False,
        }
        super_hypers.update(this_hypers)

        return super_hypers

    @staticmethod
    def default_data_path() -> str:
        return "data/ppi"

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(params, metadata=metadata, **kwargs)

        self._tied_fwd_bkwd_edge_types = get_tied_edge_types(
            tie_fwd_bkwd_edges=params["tie_fwd_bkwd_edges"], num_fwd_edge_types=1,
        )

        self._num_edge_types = compute_number_of_edge_types(
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            num_fwd_edge_types=1,
            add_self_loop_edges=params["add_self_loop_edges"],
        )

        # Things that will be filled once we load data:
        self._loaded_data: Dict[DataFold, List[PPIGraphSample]] = {}

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def node_feature_shape(self) -> Tuple:
        some_data_fold = next(iter(self._loaded_data.values()))
        return (some_data_fold[0].node_features.shape[-1],)

    @property
    def num_node_target_labels(self) -> int:
        return 121

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        # Data in format as downloaded from https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip
        # If we haven't defined what folds to load, load all:
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path, DataFold.TRAIN)
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path, DataFold.VALIDATION)
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path, DataFold.TEST)

    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        raise NotImplementedError()

    def __load_data(self, data_dir: RichPath, data_fold: DataFold) -> List[PPIGraphSample]:
        if data_fold == DataFold.TRAIN:
            data_name = "train"
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
        elif data_fold == DataFold.TEST:
            data_name = "test"
        else:
            raise ValueError("Unknown data fold '%s'" % str(data_fold))
        print(" Loading PPI %s data from %s." % (data_name, data_dir))

        graph_json_data = data_dir.join("%s_graph.json" % data_name).read_by_file_suffix()
        node_to_features = data_dir.join("%s_feats.npy" % data_name).read_by_file_suffix()
        node_to_labels = data_dir.join("%s_labels.npy" % data_name).read_by_file_suffix()
        node_to_graph_id = data_dir.join("%s_graph_id.npy" % data_name).read_by_file_suffix()

        # We read in all the data in two steps:
        #  (1) Read features and labels. Implicitly, this gives us the number of nodes per graph.
        #  (2) Read all edges, and shift them so that each graph starts with node 0.

        graph_id_to_edges: Dict[int, List[Tuple[int, int]]] = {}
        graph_id_to_features: Dict[int, List[np.ndarray]] = {}
        graph_id_to_labels: Dict[int, List[np.ndarray]] = {}
        graph_id_to_node_offset: Dict[int, int] = {}

        num_total_nodes = node_to_features.shape[0]
        for node_id in range(num_total_nodes):
            graph_id = node_to_graph_id[node_id]

            # In case we are entering a new graph, note its ID, so that we can normalise everything to start at 0
            if graph_id not in graph_id_to_edges:
                graph_id_to_edges[graph_id] = []
                graph_id_to_features[graph_id] = []
                graph_id_to_labels[graph_id] = []
                graph_id_to_node_offset[graph_id] = node_id

            graph_id_to_features[graph_id].append(node_to_features[node_id])
            graph_id_to_labels[graph_id].append(node_to_labels[node_id])

        for edge_info in graph_json_data["links"]:
            src_node, tgt_node = edge_info["source"], edge_info["target"]
            # First, shift node IDs so that each graph starts at node 0:
            graph_id = node_to_graph_id[src_node]
            graph_node_offset = graph_id_to_node_offset[graph_id]
            src_node, tgt_node = src_node - graph_node_offset, tgt_node - graph_node_offset

            graph_id_to_edges[graph_id].append((src_node, tgt_node))

        final_graphs = []
        for graph_id in graph_id_to_edges.keys():
            num_nodes = len(graph_id_to_features[graph_id])

            adjacency_lists, type_to_node_to_num_inedges = process_adjacency_lists(
                adjacency_lists=[graph_id_to_edges[graph_id]],
                num_nodes=num_nodes,
                add_self_loop_edges=self.params["add_self_loop_edges"],
                tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            )

            final_graphs.append(
                PPIGraphSample(
                    adjacency_lists=adjacency_lists,
                    type_to_node_to_num_inedges=type_to_node_to_num_inedges,
                    node_features=np.array(graph_id_to_features[graph_id]),
                    node_labels=np.array(graph_id_to_labels[graph_id]),
                )
            )

        return final_graphs

    # -------------------- Minibatching --------------------
    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "node_labels": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "node_labels": (None, None)},
        )

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[PPIGraphSample]:
        loaded_data = self._loaded_data[data_fold]
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(loaded_data)
        return iter(loaded_data)

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["node_labels"] = []
        return new_batch

    def _add_graph_to_batch(self, raw_batch, graph_sample: PPIGraphSample) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["node_labels"].append(graph_sample.node_labels)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        batch_labels["node_labels"] = np.concatenate(raw_batch["node_labels"], axis=0)
        return batch_features, batch_labels
