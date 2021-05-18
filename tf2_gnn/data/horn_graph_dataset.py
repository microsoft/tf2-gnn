from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from typing import Any, Dict, Iterable, List, Iterator, Tuple, Optional, Set
import numpy as np
import random
import tensorflow as tf
import glob
import json
import pickle
import scipy.stats as ss


class HornGraphSample(GraphSample):
    """Data structure holding a single horn graph."""
    def __init__(
            self,
            adjacency_lists: List[np.ndarray],
            node_features: np.ndarray,
            node_indices: np.ndarray,
            node_label: np.ndarray,
    ):
        super().__init__(adjacency_lists, [], node_features)
        self._node_label = node_label
        self._node_indices = node_indices

    @property
    def node_label(self) -> np.ndarray:
        """Node labels to predict as ndarray of shape [V, C]"""
        return self._node_label


class HornGraphDataset(GraphDataset[HornGraphSample]):
    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        super().__init__(params, metadata=metadata)
        self._num_edge_types:List[int]
        self._total_number_of_nodes:int
        self._node_number_per_edge_type = list()
        self._node_feature_shape: Optional[Tuple[int]]
        self._loaded_data: Dict[DataFold, List[GraphSample]] = {}
        self._label_list = {}
        self._file_list = {}
        self._benchmark = params["benchmark"]
        self.label_type = params["label_type"]
        self._graph_type = params["graph_type"]
        self._node_vocab_size = 0
        #self._read_from_pickle=False
        self._path=""
        self._json_type=".JSON"
        self._class_weight_fold={}

    def load_data(self,folds_to_load: Optional[Set[DataFold]] = None) -> None:
        '''been run automatically when create the object of this class'''
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}
            # folds_to_load = {DataFold.TRAIN}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(DataFold.TRAIN)
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(DataFold.VALIDATION)
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(DataFold.TEST)

    def __load_data(self, data_fold: DataFold) -> List[HornGraphSample]:
        if data_fold == DataFold.TRAIN:
            data_name = "train"
            self._node_number_per_edge_type = []  # reset to empty list
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
            self._node_number_per_edge_type = []
        elif data_fold == DataFold.TEST:
            data_name = "test"
            self._node_number_per_edge_type = []

        print("data_fold", data_name)
        if self.params["pickle"]==True:
            print("read GNNInputs from pickle file")
            pickle_file_name = self.label_type +"-"+self._graph_type+ "-" + self._benchmark + "-gnnInput_" + data_name + "_data"
            print(pickle_file_name)
            raw_inputs = pickleRead(pickle_file_name)


            final_graphs = raw_inputs.final_graphs
            node_num_list = []
            for g in final_graphs:
                node_num_list.append(len(g.node_features))
            # Vocabulary size should be total vocabulary size of train, valid, test data
            self._node_vocab_size = len(raw_inputs.vocabulary_set)
            # if self._node_vocab_size<max(node_num_list):
            #     self._node_vocab_size=max(node_num_list)

            self._num_edge_types = raw_inputs._num_edge_types
            self._total_number_of_nodes = raw_inputs._total_number_of_nodes
            self._node_number_per_edge_type = raw_inputs._node_number_per_edge_type
            self._label_list[data_name] = raw_inputs.labels
            self._file_list[data_name] = raw_inputs.file_names
            self._class_weight_fold[data_name]= raw_inputs.class_weight[data_name]

            print("raw_inputs.label_size", raw_inputs.label_size)
            print("raw_inputs._total_number_of_nodes", raw_inputs._total_number_of_nodes)
            print("raw_inputs._num_edge_types", raw_inputs._num_edge_types)
            print("raw_inputs._node_number_per_edge_type", raw_inputs._node_number_per_edge_type)
            print("_node_vocab_size",self._node_vocab_size )
            print("raw_inputs.class_weight", self._class_weight_fold)

        else:
            tokenized_node_label_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            adjacency_lists = [
                np.array([[0, 1], [0, 2], [1, 3], [1, 4], [1, 5], [4, 7], [4, 8], [2, 6], [6, 9], [6, 10]])]
            node_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            learning_labels = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
            self._num_edge_types = 1
            self._node_number_per_edge_type=[2]
            self._total_number_of_nodes = len(tokenized_node_label_ids)
            self._label_list[data_name] = learning_labels
            self._file_list[data_name] = "tree_leaf_identify"
            self._class_weight_fold[data_name] = {'weight_for_0': 1, 'weight_for_1': 1}
            self._node_vocab_size = 1
            final_graphs = [HornGraphSample(
                adjacency_lists=adjacency_lists,
                node_features=np.array(tokenized_node_label_ids),  # node_label_ids,tokenized_node_label_ids
                node_indices=np.array(node_indices),
                node_label=np.array(learning_labels)
            )]

        return final_graphs

    def load_data_from_list(self):
        raise NotImplementedError()
        pass

    @property
    def node_feature_shape(self) -> Tuple:
        """Return the shape of the node features."""
        some_data_fold = next(iter(self._loaded_data.values()))
        return (some_data_fold[0].node_features.shape[-1],)

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def total_number_of_nodes(self) -> int:
        return self._total_number_of_nodes

    # -------------------- Minibatching --------------------
    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        batch_features_types = {
            "node_features": tf.int32,
            "node_to_graph_map": tf.int32,
            "num_graphs_in_batch": tf.int32,
            "label_node_indices": tf.int32
            # "current_node_index":tf.int32
        }
        # print("self.node_feature_shape",self.node_feature_shape)
        batch_features_shapes = {
            "node_features": (None,),  # + self.node_feature_shape,  #no offset in minibatch
            "node_to_graph_map": (None,),
            "num_graphs_in_batch": (),
            "label_node_indices": (None,)
            # "current_node_index":(None,)
        }
        for edge_type_idx, edge_number in enumerate(self._node_number_per_edge_type):
            batch_features_types[f"adjacency_list_{edge_type_idx}"] = tf.int32
            batch_features_shapes[f"adjacency_list_{edge_type_idx}"] = (None, edge_number)

        if self.label_type=="argument_bound":
            return GraphBatchTFDataDescription(
                batch_features_types=batch_features_types,
                batch_features_shapes=batch_features_shapes,
                batch_labels_types={**data_description.batch_labels_types, "node_labels": tf.float32},
                batch_labels_shapes={**data_description.batch_labels_shapes, "node_labels": (None,2)})
        else:
            return GraphBatchTFDataDescription(
                batch_features_types=batch_features_types,
                batch_features_shapes=batch_features_shapes,
                batch_labels_types={**data_description.batch_labels_types, "node_labels": tf.float32},
                batch_labels_shapes={**data_description.batch_labels_shapes, "node_labels": (None,)},)

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[HornGraphSample]:
        loaded_data = self._loaded_data[data_fold]
        if data_fold == DataFold.TRAIN:
            random.shuffle(loaded_data)
        return iter(loaded_data)

    def _new_batch(self) -> Dict[str, Any]:
        # new_batch = super()._new_batch()
        # new_batch["node_argument"]=[]
        # new_batch["node_labels"] = []
        return {
            "node_features": [],
            "adjacency_lists": [[] for _ in range(self.num_edge_types)],
            "node_to_graph_map": [],
            "num_graphs_in_batch": 0,
            "num_nodes_in_batch": 0,
            "label_node_indices": [],
            "node_labels": []
            # "current_node_index":[]
        }
        return new_batch

    def _add_graph_to_batch(self, raw_batch, graph_sample: HornGraphSample) -> None:
        # super()._add_graph_to_batch(raw_batch, graph_sample)
        num_nodes_in_graph = len(graph_sample.node_features)
        # print("----add new batch---")
        offset = raw_batch["num_nodes_in_batch"]
        # print("num_nodes_in_graph",num_nodes_in_graph)
        # print("offset",offset)
        # raw_batch["node_features"].extend(graph_sample.node_features+offset)
        raw_batch["node_features"].extend(graph_sample.node_features)

        raw_batch["node_to_graph_map"].append(
            np.full(
                shape=[num_nodes_in_graph],
                fill_value=raw_batch["num_graphs_in_batch"],
                dtype=np.int32,
            )
        )
        # print("len(raw bach adjacent list)",len(raw_batch["adjacency_lists"]))
        # print("sample adjacent list",len(graph_sample.adjacency_lists))

        for edge_type_idx, (batch_adjacency_list, sample_adjacency_list) in enumerate(
                zip(raw_batch["adjacency_lists"], graph_sample.adjacency_lists)):

            if len(sample_adjacency_list) != 0:
                edge_number = sample_adjacency_list.shape[1]
                batch_adjacency_list.append(
                    graph_sample.adjacency_lists[edge_type_idx].reshape(-1, edge_number)
                    + offset )
            # print("graph_sample.adjacency_lists",graph_sample.adjacency_lists[edge_type_idx] + offset)

        raw_batch["label_node_indices"].extend(graph_sample._node_indices + offset)
        raw_batch["node_labels"].extend(graph_sample._node_label)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features: Dict[str, Any] = {}
        #if two continuous graphs' nodes > max_nodes_per_batch, then raw_batch["node_labels"] will be empty and finally pose "train_loss=nan"
        batch_labels: Dict[str, Any] = {"node_labels": raw_batch["node_labels"]}
        batch_features["node_features"] = np.array(raw_batch["node_features"])
        # print("raw_batch node_to_graph_map len",len(raw_batch["node_to_graph_map"]))
        if len(raw_batch["node_to_graph_map"]) == 0:
            batch_features["node_to_graph_map"] = raw_batch["node_to_graph_map"]
        else:
            batch_features["node_to_graph_map"] = np.concatenate(raw_batch["node_to_graph_map"])
        batch_features["num_graphs_in_batch"] = raw_batch["num_graphs_in_batch"]
        for i, (adjacency_list, edge_num) in enumerate(
                zip(raw_batch["adjacency_lists"], self._node_number_per_edge_type)):
            if len(adjacency_list) > 0:
                batch_features[f"adjacency_list_{i}"] = np.concatenate(adjacency_list)
            else:
                batch_features[f"adjacency_list_{i}"] = np.zeros(shape=(0, edge_num), dtype=np.int32)
                # batch_features[f"adjacency_list_{1}"] = np.zeros(shape=(0, 3),dtype=np.int32)

        # batch_features, batch_labels = super()._finalise_batch(raw_batch)
        batch_features["label_node_indices"] = raw_batch["label_node_indices"]
        return batch_features, batch_labels



def pickleRead(pickle_file_name, path=""):
    file = path + '../pickleData/' + pickle_file_name + '.txt'
    print('pickle read ' + file)
    with open(file, "rb") as fp:
        content = pickle.load(fp)
    return content