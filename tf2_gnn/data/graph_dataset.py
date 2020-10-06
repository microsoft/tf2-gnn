from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Iterator, Tuple, TypeVar, Generic, NamedTuple, Set, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath, DoubleBufferedIterator


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class GraphBatchTFDataDescription(NamedTuple):
    batch_features_types: Dict[str, Any]
    batch_features_shapes: Dict[str, Any]
    batch_labels_types: Dict[str, Any]
    batch_labels_shapes: Dict[str, Any]


class GraphSample(object):
    """Data structure holding information for a single graph."""

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_inedges: np.ndarray,
        node_features: np.ndarray,
    ):
        super().__init__()
        self._adjacency_lists = adjacency_lists
        self._type_to_node_to_num_inedges = type_to_node_to_num_inedges
        self._node_features = node_features

    @property
    def adjacency_lists(self) -> List[np.ndarray]:
        """Adjacency information by edge type as list of ndarrays of shape [E, 2]"""
        return self._adjacency_lists

    @property
    def type_to_node_to_num_inedges(self) -> np.ndarray:
        """Number of incoming edge by edge type as ndarray of shape [V]"""
        return self._type_to_node_to_num_inedges

    @property
    def node_features(self) -> np.ndarray:
        """Initial node features as ndarray of shape [V, ...]"""
        return self._node_features


GraphSampleType = TypeVar("GraphSampleType", bound=GraphSample)


class GraphDataset(Generic[GraphSampleType]):
    """Abstract class turning graph datasets into iterators providing minibatches of graph
    samples.
    This class implements the core logic for batching of graphs into one huge graph with
    disconnected components, and provides hooks to insert additional data.

    To use this for your own data type, graph samples need to be stored as datapoints
    extending the GraphSample class.
    Besides implementing the abstract methods, implementors should override _new_batch,
    _add_graph_to_batch, and _finalise_batch to add dataset-specific things (such as
    labels for the graphs or nodes) to the generated minibatches.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "max_nodes_per_batch": 10000,
        }

    def __init__(
        self,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        use_worker_threads: bool = True,
    ):
        self._params = params
        self._metadata = metadata if metadata is not None else {}
        self._use_worker_threads = use_worker_threads

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    @abstractmethod
    def num_edge_types(self) -> int:
        pass

    @property
    @abstractmethod
    def node_feature_shape(self) -> Tuple:
        pass

    @abstractmethod
    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        pass

    @abstractmethod
    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        pass

    @abstractmethod
    def _graph_iterator(self, data_fold: DataFold) -> Iterator[GraphSampleType]:
        """Returns iterator over data in specified fold of the dataset.
        Note: The iterator is expected to shuffle training data on every call.
        """
        pass

    def graph_batch_iterator(
        self, data_fold: DataFold
    ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Get the graph batch iterator.

        Graph are batched into a single graph with multiple disconnected components.
        The "batch" is the full graph, and each disconnected component is one of the
        sample graphs that make up the batch.

        We use the following abbreviations in shape descriptions:
        * V: number of nodes
        * D: state dimension
        * L: number of different edge types
        * E: number of edges of a given edge type
        * G: number of graphs in the batch

        Returns:
            An iterator which outputs a single minibatch as a pair of dictionaries
            representing features (i.e., the graphs) and labels (e.g., regression
            targets).

            The features dictionary has at least these three entries (which are
            consumed by the "general" infrastructure):
                * node_features: a numpy float32 array of shape [V, ...] representing the
                    features for each node in the graph. Usually a one-hot-encoding of
                    the atom type, tokens in label, etc.
                * node_to_graph_map: a numpy int32 array of shape [V] that represents
                    which graph in the batch each node comes from.
                * adjacency_lists: a list of length L of numpy int32 arrays of shape
                    [E, 2] which represents an adjacency list for a given edge type.
                    Concretely, adjacency_lists[l][k,:] == [v, u]
                    means that the k-th edge of type l connects node v to node u.
        """
        graph_sample_iterator = self._graph_iterator(data_fold)
        for graph_batch in self.graph_batch_iterator_from_graph_iterator(graph_sample_iterator):
            yield graph_batch

    def graph_batch_iterator_from_graph_iterator(
        self, graph_sample_iterator: Iterator[GraphSampleType]
    ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        raw_batch = self._new_batch()
        for graph_sample in graph_sample_iterator:
            num_nodes_in_graph = len(graph_sample.node_features)

            # Yield the batch if adding the current graph_sample would make it too big.
            if self._batch_would_be_too_full(raw_batch, graph_sample):
                yield self._finalise_batch(raw_batch)
                raw_batch = self._new_batch()

            # Extend the bath with the current graph_sample.
            self._add_graph_to_batch(raw_batch, graph_sample)

            # Increment counters.
            raw_batch["num_graphs_in_batch"] += 1
            raw_batch["num_nodes_in_batch"] += num_nodes_in_graph

        yield self._finalise_batch(raw_batch)

    def _batch_would_be_too_full(
        self, raw_batch: Dict[str, Any], graph_sample: GraphSampleType
    ) -> bool:
        """Return whether the current raw batch would be too full if graph_sample was added."""
        num_nodes_in_graph = len(graph_sample.node_features)
        return (
            raw_batch["num_nodes_in_batch"] + num_nodes_in_graph
            > self._params["max_nodes_per_batch"]
        )

    def _new_batch(self) -> Dict[str, Any]:
        """Return a dictionary suitable for collecting data for a fresh minibatch."""
        return {
            "node_features": [],
            "adjacency_lists": [[] for _ in range(self.num_edge_types)],
            "node_to_graph_map": [],
            "num_graphs_in_batch": 0,
            "num_nodes_in_batch": 0,
        }

    def _add_graph_to_batch(self, raw_batch: Dict[str, Any], graph_sample: GraphSampleType) -> None:
        """Add a graph sample to a minibatch under preparation.

        Args:
            raw_batch: Holder for the currently constructed minibatch (created by _new_batch)
            graph_sample: Graph sample to add.
        """
        num_nodes_in_graph = len(graph_sample.node_features)
        raw_batch["node_features"].extend(graph_sample.node_features)
        raw_batch["node_to_graph_map"].append(
            np.full(
                shape=[num_nodes_in_graph],
                fill_value=raw_batch["num_graphs_in_batch"],
                dtype=np.int32,
            )
        )
        for edge_type_idx, batch_adjacency_list in enumerate(raw_batch["adjacency_lists"]):
            batch_adjacency_list.append(
                graph_sample.adjacency_lists[edge_type_idx].reshape(-1, 2)
                + raw_batch["num_nodes_in_batch"]
            )

    def _finalise_batch(self, raw_batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Turns a raw batch into a minibatch ready to be fed to the model (i.e., converts
        lists to numpy arrays, and concatenate appropriately.

        Args:
            raw_batch: Holder for the currently constructed minibatch (created by _new_batch,
                filled by repeated _add_graph_to_batch calls)

        Returns:
            Pair of batch_features and batch_labels, for use in model.
        """
        batch_features: Dict[str, Any] = {}
        batch_labels: Dict[str, Any] = {}
        batch_features["node_features"] = np.array(raw_batch["node_features"])
        batch_features["node_to_graph_map"] = np.concatenate(raw_batch["node_to_graph_map"])
        batch_features["num_graphs_in_batch"] = raw_batch["num_graphs_in_batch"]
        for i, adjacency_list in enumerate(raw_batch["adjacency_lists"]):
            if len(adjacency_list) > 0:
                batch_features[f"adjacency_list_{i}"] = np.concatenate(adjacency_list)
            else:
                batch_features[f"adjacency_list_{i}"] = np.zeros(shape=(0, 2), dtype=np.int32)

        return batch_features, batch_labels

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        """Provides information about dataset-specific data shapes and types
        in the generated minibatch (i.e., additional keys in the features or
        label dictionaries).
        """
        batch_features_types = {
            "node_features": tf.float32,
            "node_to_graph_map": tf.int32,
            "num_graphs_in_batch": tf.int32,
        }
        batch_features_shapes = {
            "node_features": (None,) + self.node_feature_shape,
            "node_to_graph_map": (None,),
            "num_graphs_in_batch": (),
        }
        for edge_type_idx in range(self.num_edge_types):
            batch_features_types[f"adjacency_list_{edge_type_idx}"] = tf.int32
            batch_features_shapes[f"adjacency_list_{edge_type_idx}"] = (None, 2)
        batch_labels_types: Dict[str, Any] = {}
        batch_labels_shapes: Dict[str, Any] = {}

        return GraphBatchTFDataDescription(
            batch_features_types=batch_features_types,
            batch_features_shapes=batch_features_shapes,
            batch_labels_types=batch_labels_types,
            batch_labels_shapes=batch_labels_shapes,
        )

    def get_tensorflow_dataset(
        self, data_fold: DataFold, use_worker_threads: Optional[bool] = None
    ) -> tf.data.Dataset:
        """Construct a TensorFlow dataset from the _graph_batch_iterator of this class.

        Returns:
            A tensorflow Dataset object. Each element in the dataset is a pair of
            dictionaries representing features and labels.
            The content of these is determined by the _finalise_batch method.
        """
        data_description = self.get_batch_tf_data_description()

        # Use class default if not explicitly specified:
        if use_worker_threads is None:
            use_worker_threads = self._use_worker_threads

        if use_worker_threads:
            graph_batch_iterator = lambda: DoubleBufferedIterator(
                self.graph_batch_iterator(data_fold)
            )
        else:
            graph_batch_iterator = lambda: self.graph_batch_iterator(data_fold)

        dataset = tf.data.Dataset.from_generator(
            generator=graph_batch_iterator,
            output_types=(
                data_description.batch_features_types,
                data_description.batch_labels_types,
            ),
            output_shapes=(
                data_description.batch_features_shapes,
                data_description.batch_labels_shapes,
            ),
        )

        return dataset
