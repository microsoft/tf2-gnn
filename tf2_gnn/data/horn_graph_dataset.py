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
            node_argument: np.ndarray = [],
            current_node_index: np.ndarray = [],
            node_control_location: np.ndarray = []
    ):
        super().__init__(adjacency_lists, [], node_features)
        self._node_label = node_label
        self._node_indices = node_indices
        self._node_argument = node_argument
        self._node_control_location = node_control_location
        self._current_node_index = current_node_index

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
        self._argument_scores = {}
        self._label_list = {}
        self._ranked_argument_scores = {}
        self._file_list = {}
        self._benchmark = params["benchmark"]
        self.label_type = params["label_type"]
        self._node_vocab_size = 0
        self._read_from_pickle=False
        self._path=""
        self._json_type=".JSON"

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
        if self._read_from_pickle==True:
            print("read GNNInputs from pickle file")
            pickle_file_name = self.label_type + "-" + self._benchmark + "-gnnInput_" + data_name + "_data"
            print(pickle_file_name)
            raw_inputs = pickleRead(pickle_file_name)
        else:
            raw_inputs=form_gnn_inputs(data_fold=[data_name], label=self.label_type, path=self._path,
                            file_type=".smt2", json_type=self._json_type)[data_name]

        final_graphs = raw_inputs.final_graphs
        node_num_list = []
        for g in final_graphs:
            node_num_list.append(len(g.node_features))
        # Vocabulary size should be total vocabulary size of train, valid, test data
        self._node_vocab_size = len(raw_inputs.vocabulary_set)
        # if self._node_vocab_size<max(node_num_list):
        #     self._node_vocab_size=max(node_num_list)

        print("raw_inputs.label_size", raw_inputs.label_size)
        print("raw_inputs._total_number_of_nodes", raw_inputs._total_number_of_nodes)
        self._total_number_of_nodes = raw_inputs._total_number_of_nodes
        print("raw_inputs._num_edge_types", raw_inputs._num_edge_types)
        self._num_edge_types = raw_inputs._num_edge_types
        print("raw_inputs._node_number_per_edge_type", raw_inputs._node_number_per_edge_type)
        self._node_number_per_edge_type = raw_inputs._node_number_per_edge_type
        self._label_list[data_name] = raw_inputs.labels
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

        return GraphBatchTFDataDescription(
            batch_features_types=batch_features_types,
            batch_features_shapes=batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "node_labels": tf.float32},
            batch_labels_shapes={**data_description.batch_labels_shapes, "node_labels": (None,)},

        )

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
        # raw_batch["current_node_index"].extend(graph_sample._current_node_index)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features: Dict[str, Any] = {}
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
        # batch_features["current_node_index"] = raw_batch["current_node_index"]
        return batch_features, batch_labels


def pickleRead(pickle_file_name, path=""):
    file = path + '../pickleData/' + pickle_file_name + '.txt'
    print('pickle read ' + file)
    with open(file, "rb") as fp:
        content = pickle.load(fp)
    return content


# form gnn inputs
def form_gnn_inputs(data_fold=["train", "valid", "test"], label="predicate_occurrence_in_clauses", path="../", file_type=".smt2",json_type=".JSON"):
    vocabulary_set, token_map = build_vocabulary(datafold=["train", "valid", "test"], path=path,json_type=json_type)
    gnn_input_fold = {}
    for df in data_fold:
        print("read graph from json file:", df)
        graphs_node_label_ids = []
        graphs_node_symbols = []
        graphs_argument_indices = []
        graphs_adjacency_lists = []
        graphs_argument_scores = []
        graphs_control_location_indices = []
        graphs_predicate_indices = []
        graphs_learning_labels = []
        total_number_of_node = 0
        file_type = file_type
        file_name_list = []
        # read from JSON
        suffix = file_type
        for fileGraph in sorted(glob.glob(path + df + "_data/" + '*' + suffix + json_type)):
            fileName = fileGraph[:fileGraph.find(suffix + json_type) + len(suffix)]
            fileName = fileName[fileName.rindex("/") + 1:]
            # read graph
            with open(fileGraph) as f:
                loaded_graph = json.load(f)
                # debug check all field if equal to empty
                if len(loaded_graph["nodeIds"]) == 0:
                    print("nodeIds==0", fileName)

                else:
                    file_name_list.append(fileGraph[:fileGraph.find(json_type)])
                    graphs_node_label_ids.append(loaded_graph["nodeIds"])
                    graphs_node_symbols.append(loaded_graph["nodeSymbolList"])
                    # read label
                    if label == "predicate_occurrence_in_clauses":
                        graphs_predicate_indices.append(loaded_graph["predicateIndices"])
                        graphs_learning_labels.append(loaded_graph["predicateOccurrenceInClause"])
                    elif label == "predicate_occurrence_in_SCG":
                        graphs_predicate_indices.append(loaded_graph["predicateIndices"])
                        graphs_learning_labels.append(loaded_graph["predicateStrongConnectedComponent"])
                    elif label == "control_location_identify":
                        graphs_control_location_indices.append(loaded_graph["controlLocationIndices"])
                    else:
                        graphs_argument_indices.append(loaded_graph["argumentIndices"])
                        # read argument from JSON file
                        parsed_arguments = parseArgumentsFromJson(loaded_graph["argumentIDList"],
                                                                  loaded_graph["argumentNameList"],
                                                                  loaded_graph["argumentOccurrence"])
                        graphs_argument_scores.append([int(argument.score) for argument in parsed_arguments])
                        graphs_control_location_indices.append(loaded_graph["controlLocationIndices"])

                    if json_type == ".hyperEdgeHornGraph.JSON":  # read adjacency_lists
                        # for hyperedge horn graph
                        graphs_adjacency_lists.append([
                            np.array(loaded_graph["argumentEdges"]),
                            np.array(loaded_graph["guardASTEdges"]),
                            np.array(loaded_graph["dataFlowASTEdges"]),
                            np.array(loaded_graph["binaryAdjacentList"]),
                            np.array(loaded_graph["controlFlowHyperEdges"]),
                            np.array(loaded_graph["dataFlowHyperEdges"]),
                            np.array(loaded_graph["ternaryAdjacencyList"])
                        ])

                    else:
                        # for layer horn graph
                        graphs_adjacency_lists.append([
                            np.array(loaded_graph["binaryAdjacentList"]),
                            np.array(loaded_graph["predicateArgumentEdges"]),
                            np.array(loaded_graph["predicateInstanceEdges"]),
                            np.array(loaded_graph["argumentInstanceEdges"]),
                            np.array(loaded_graph["controlHeadEdges"]),
                            np.array(loaded_graph["controlBodyEdges"]),
                            np.array(loaded_graph["controlArgumentEdges"]),
                            np.array(loaded_graph["guardEdges"]),
                            np.array(loaded_graph["dataEdges"])
                            # np.array(loaded_graph["unknownEdges"])
                        ])
                    total_number_of_node += len(loaded_graph["nodeIds"])
        # form label
        if label == "predicate_occurrence_in_clauses" or label == "predicate_occurrence_in_SCG":
            gnn_input_fold[df] = form_predicate_occurrence_related_label_graph_sample(graphs_node_label_ids,
                                                                                      graphs_node_symbols,
                                                                                      graphs_adjacency_lists,
                                                                                      total_number_of_node,
                                                                                      vocabulary_set, token_map,
                                                                                      file_name_list,
                                                                                      graphs_predicate_indices,
                                                                                      graphs_learning_labels)
        else:
            gnn_input_fold[df] = form_argument_occurrence_related_label_graph_sample(graphs_node_label_ids,
                                                                                     graphs_node_symbols,
                                                                                     graphs_argument_indices,
                                                                                     graphs_adjacency_lists,
                                                                                     graphs_argument_scores,
                                                                                     total_number_of_node,
                                                                                     graphs_control_location_indices,
                                                                                     file_name_list, label,
                                                                                     vocabulary_set, token_map)
    return gnn_input_fold


def build_vocabulary(datafold=["train", "valid", "test"], path="", json_type=".layerHornGraph.JSON"):
    vocabulary_set = set(["unknown"])
    for fold in datafold:
        for json_file in glob.glob(path + fold + "_data/*" + json_type):
            with open(json_file) as f:
                loaded_graph = json.load(f)
                vocabulary_set.update(loaded_graph["nodeSymbolList"])
    token_map = {}
    token_id = 0
    for word in vocabulary_set:
        token_map[word] = token_id
        token_id = token_id + 1
    return vocabulary_set, token_map
class raw_graph_inputs():
    def __init__(self, num_edge_types, total_number_of_nodes):
        self._num_edge_types = num_edge_types
        self._total_number_of_nodes = total_number_of_nodes
        self._node_number_per_edge_type = []
        self.final_graphs = None
        self.argument_scores = []
        self.labels = []
        self.ranked_argument_scores = []
        self.file_names = []
        self.argument_identify = []
        self.control_location_identify = []
        self.label_size = 0
        self.vocabulary_set = set()
        self.token_map = {}
def parseArgumentsFromJson(id_list, name_list, occurrence_list):
    parsed_argument_list = []
    for id, name, occurrence in zip(id_list, name_list, occurrence_list):
        head = name[:name.rfind(":")]
        hint = name[name.rfind(":") + 1:]
        parsed_argument_list.append(ArgumentInfo(id, head, hint, occurrence))
    return parsed_argument_list
class ArgumentInfo:
    def __init__(self, ID, head, arg, score):
        self.ID = ID
        self.head = head
        self.arg = arg
        self.score = score
        self.nodeUniqueIDInGraph = -1
        self.nodeLabelUniqueIDInGraph = -1
def form_predicate_occurrence_related_label_graph_sample(graphs_node_label_ids, graphs_node_symbols,
                                                         graphs_adjacency_lists, total_number_of_node,
                                                         vocabulary_set, token_map, file_name_list,
                                                         graphs_predicate_indices, graphs_learning_labels):
    final_graphs = []
    raw_data_graph = get_batch_graph_sample_info(graphs_adjacency_lists, total_number_of_node, vocabulary_set,
                                                 token_map)
    for node_label_ids, node_symbols, adjacency_lists, file_name, predicate_indices, learning_labels in zip(
            graphs_node_label_ids, graphs_node_symbols,
            graphs_adjacency_lists,
            file_name_list,
            graphs_predicate_indices,
            graphs_learning_labels):
        raw_data_graph.file_names.append(file_name)
        # node tokenization
        tokenized_node_label_ids = []
        for symbol in node_symbols:
            tokenized_node_label_ids.append(token_map[symbol])
        raw_data_graph.labels.append(learning_labels)

        final_graphs.append(
            HornGraphSample(
                adjacency_lists=(adjacency_lists),
                node_features=np.array(tokenized_node_label_ids),
                node_indices=np.array(predicate_indices),
                node_label=np.array(learning_labels),
            )
        )

        raw_data_graph.label_size += len(learning_labels)
    raw_data_graph.final_graphs = final_graphs.copy()
    return raw_data_graph


def form_argument_occurrence_related_label_graph_sample(graphs_node_label_ids, graphs_node_symbols,
                                                        graphs_argument_indices, graphs_adjacency_lists,
                                                        graphs_argument_scores, total_number_of_node,
                                                        graphs_control_location_indices,
                                                        file_name_list,
                                                        label, vocabulary_set, token_map):
    final_graphs_v1 = []
    raw_data_graph = get_batch_graph_sample_info(graphs_adjacency_lists, total_number_of_node, vocabulary_set,
                                                 token_map)
    total_label = 0
    total_nodes = 0
    if len(graphs_control_location_indices) == 0:
        graphs_control_location_indices = graphs_argument_indices
    for node_label_ids, node_symbols, argument_indices, adjacency_lists, argument_scores, control_location_indices, \
        file_name in zip(graphs_node_label_ids, graphs_node_symbols, graphs_argument_indices, graphs_adjacency_lists,
                         graphs_argument_scores, graphs_control_location_indices, file_name_list):
        # convert to rank
        ranked_argument_scores = ss.rankdata(argument_scores, method="dense")
        argument_identify = np.array([0] * len(node_label_ids))
        argument_identify[argument_indices] = 1
        total_nodes += len(node_label_ids)
        # total_label += len(argument_indices)
        control_location_identify = np.array([0] * len(node_label_ids))
        control_location_identify[control_location_indices] = 1

        raw_data_graph.argument_identify.append(argument_identify)
        raw_data_graph.control_location_identify.append(control_location_identify)
        raw_data_graph.ranked_argument_scores.append(ranked_argument_scores)
        raw_data_graph.argument_scores.append(argument_scores)
        raw_data_graph.file_names.append(file_name)

        # node tokenization
        tokenized_node_label_ids = []
        for symbol in node_symbols:
            tokenized_node_label_ids.append(token_map[symbol])

        if label == "rank":
            raw_data_graph.labels.append(argument_scores)
            total_label += len(ranked_argument_scores)
            final_graphs_v1.append(
                HornGraphSample(
                    adjacency_lists=(adjacency_lists),
                    node_features=np.array(tokenized_node_label_ids),
                    node_label=np.array(ranked_argument_scores),
                    node_indices=np.array(argument_indices),
                )
            )
            raw_data_graph.label_size += len(ranked_argument_scores)
        elif label == "occurrence":
            raw_data_graph.labels.append(argument_scores)
            total_label += len(argument_scores)
            final_graphs_v1.append(
                HornGraphSample(
                    adjacency_lists=(adjacency_lists),
                    node_features=np.array(tokenized_node_label_ids),
                    node_label=np.array(argument_scores),
                    node_indices=np.array(argument_indices),
                )
            )
            raw_data_graph.label_size += len(argument_scores)
        elif label == "argument_identify":
            raw_data_graph.labels.append(argument_identify)
            total_label += len(argument_identify)
            final_graphs_v1.append(
                HornGraphSample(
                    adjacency_lists=(adjacency_lists),
                    node_features=np.array(tokenized_node_label_ids),
                    node_indices=np.array(argument_indices),
                    node_label=np.array(argument_identify)
                )
            )
            raw_data_graph.label_size += len(argument_identify)
        elif label == "control_location_identify":
            raw_data_graph.labels.append(control_location_identify)
            total_label += len(control_location_identify)
            final_graphs_v1.append(
                HornGraphSample(
                    adjacency_lists=(adjacency_lists),
                    node_features=np.array(tokenized_node_label_ids),
                    node_label=np.array(control_location_identify),
                )
            )
            raw_data_graph.label_size += len(control_location_identify)
        else:
            pass
    raw_data_graph.final_graphs = final_graphs_v1.copy()
    print("total_label", total_label)
    print("total_nodes", total_nodes)
    return raw_data_graph


def get_batch_graph_sample_info(graphs_adjacency_lists, total_number_of_node, vocabulary_set, token_map):
    raw_data_graph = raw_graph_inputs(len(graphs_adjacency_lists[0]),
                                      total_number_of_node)  # graphs_adjacency_lists[0] means the first graph's adjacency_list
    temp_graph_index = 0
    for i, graphs_adjacency in enumerate(graphs_adjacency_lists):
        temp_count = 0
        for edge_type in graphs_adjacency:
            if len(edge_type) != 0:
                temp_count = temp_count + 1
        if temp_count == len(graphs_adjacency):
            temp_graph_index = i
    for edge_type in graphs_adjacency_lists[temp_graph_index]:
        # if len(edge_type)!=0:
        raw_data_graph._node_number_per_edge_type.append(len(edge_type[0]))

    raw_data_graph.vocabulary_set = vocabulary_set
    raw_data_graph.token_map = token_map
    return raw_data_graph
