"""Message passing layer."""
from abc import abstractmethod
from typing import Dict, List, NamedTuple, Tuple, Any

import tensorflow as tf

from tf2_gnn.utils.param_helpers import (
    get_activation_function,
    get_aggregation_function,
)


class MessagePassingInput(NamedTuple):
    """A named tuple to hold input to the message passing layer."""

    node_embeddings: tf.Tensor
    adjacency_lists: Tuple[tf.Tensor, ...]


class MessagePassing(tf.keras.layers.Layer):
    """Abstract class to compute new graph states by neural message passing.

    Users should create a specific type of message passing layer by:
        * Implementing the abstract method `_message_function`, which performs the calculation of
          the messages that should be sent around the graph.
        * (Optional) Overriding the `_aggregation_function` method, which calculate the new (pre-
          activation) node state by aggregating the messages that were sent to the node. The built
          in defaults for this method are sum, mean, max and sqrt_n, which can be chosen by setting
          the "aggregation_function" setting the parameter dictionary.


    Throughout we use the following abbreviations in shape descriptions:
        * V: number of nodes
        * D: state dimension
        * L: number of different edge types
        * E: number of edges of a given edge type
        * D: input node representation dimension
        * H: output node representation dimension (set as hidden_dim)
    """

    @classmethod
    def get_default_hyperparameters(cls):
        return {
            "aggregation_function": "sum",  # One of sum, mean, max, sqrt_n
            "message_activation_function": "relu",  # One of relu, leaky_relu, elu, gelu, tanh
            "message_activation_before_aggregation": False,  # Change to True to apply activation _before_ aggregation.
            "hidden_dim": 7,
        }

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._hidden_dim = int(params["hidden_dim"])

        aggregation_fn_name = params["aggregation_function"]
        self._aggregation_fn = get_aggregation_function(aggregation_fn_name)

        self._message_activation_before_aggregation = params.get(
            "message_activation_before_aggregation", False
        )

        activation_fn_name = params["message_activation_function"]
        self._activation_fn = get_activation_function(activation_fn_name)

    @abstractmethod
    def _message_function(
        self,
        edge_source_states: tf.Tensor,
        edge_target_states: tf.Tensor,
        num_incoming_to_node_per_message: tf.Tensor,
        edge_type_idx: int,
        training: bool,
    ) -> tf.Tensor:
        """Abstract method to calculate the messages passed from a source nodes to target nodes.

        NOTE: This function should calculate messages for a single edge type.

        Args:
            edge_source_states: A tensor of shape [E, D] giving the node states at the source node
                of each edge.
            edge_target_states: A tensor of shape [E, D] giving the node states at the target node
                of each edge.
            num_incoming_to_node_per_message: A tensor of shape [E] giving the number of messages
                entering the target node of each edge. For example, if
                num_incoming_to_node_per_message[i] = 4, then there are 4 messages whose target
                node is the target node of message i.
            edge_type_idx: The edge type that that these messages correspond to.

        Returns:
            If this layer is to be used with the `_compute_new_node_embeddings` function as defined
            in this class, this function should return a tensor of shape [E, H] representing the
            messages passed along each edge.
        """
        pass

    def call(self, inputs: MessagePassingInput, training: bool = False):
        """Call the message passing layer.

        Args:
            inputs: A tuple containing two items:
                node_embeddings: float32 tensor of shape [V, D], the original representation of each
                    node in the graph.
                adjacency_lists: Tuple of L adjacency lists, represented as int32 tensors of shape
                    [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
                    of type l connects node v to node u.
            training: A bool that denotes whether we are in training mode.

        Returns:
            float32 tensor of shape [V, hidden_dim]
        """
        node_embeddings, adjacency_lists = (
            inputs.node_embeddings,
            inputs.adjacency_lists,
        )
        num_nodes = tf.shape(node_embeddings)[0]

        messages_per_type = self._calculate_messages_per_type(
            adjacency_lists, node_embeddings, training
        )

        edge_type_to_message_targets = [
            adjacency_list_for_edge_type[:, 1]
            for adjacency_list_for_edge_type in adjacency_lists
        ]

        new_node_states = self._compute_new_node_embeddings(
            node_embeddings,
            messages_per_type,
            edge_type_to_message_targets,
            num_nodes,
            training,
        )  # Shape [V, H]

        return new_node_states

    def _compute_new_node_embeddings(
        self,
        cur_node_embeddings: tf.Tensor,
        messages_per_type: List[tf.Tensor],
        edge_type_to_message_targets: List[tf.Tensor],
        num_nodes: tf.Tensor,
        training: bool,
    ):
        """Aggregate the messages using the aggregation function specified in the params dict.

        If more exotic types of aggregation are required (such as attention methods, etc.), this
        method should be overwritten in the child class.

        NOTE: If you are overriding this method definition, the `messages` input is a list
        containing the return value of the `_message_function` method for each edge type.

        Args:
            cur_node_embeddings: Current node embeddings as a float32 tensor of shape [V, H]
            messages_per_type: A list of messages to be aggregated with length L. Element i of the
            list should be a tensor of the messages for edge type i, with shape [E, H]
            edge_type_to_message_targets: A list of tensors containing the target nodes of each
                message, for each edge type. Each tensor in the list should have shape [E]. For
                example, edge_type_to_message_targets[i][j] = v means that the jth message of edge
                type i is being sent to node v.
            num_nodes: The total number of nodes in the graph, V.

        Returns:
            A tensor of shape [V, H] representing the aggregated messages for each node.

        """
        # Let M be the number of messages (sum of all E):
        message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]
        messages = tf.concat(messages_per_type, axis=0)  # Shape [M, H]

        if self._message_activation_before_aggregation:
            messages = self._activation_fn(messages)  # Shape [M, H]

        aggregated_messages = self._aggregation_fn(
            data=messages, segment_ids=message_targets, num_segments=num_nodes
        )  # Shape [V, H]

        if not self._message_activation_before_aggregation:
            aggregated_messages = self._activation_fn(aggregated_messages)  # Shape [V, H]

        return aggregated_messages

    def _calculate_messages_per_type(
        self,
        adjacency_lists: Tuple[tf.Tensor, ...],
        node_embeddings: tf.Tensor,
        training: bool = False,
    ) -> List[tf.Tensor]:
        messages_per_type = []  # list of tensors of messages of shape [E, H]

        # Calculate this once.
        type_to_num_incoming_edges = calculate_type_to_num_incoming_edges(
            node_embeddings, adjacency_lists
        )
        # Collect incoming messages per edge type
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = adjacency_list_for_edge_type[:, 0]
            edge_targets = adjacency_list_for_edge_type[:, 1]
            edge_source_states = tf.nn.embedding_lookup(
                params=node_embeddings, ids=edge_sources
            )  # Shape [E, H]
            edge_target_states = tf.nn.embedding_lookup(
                params=node_embeddings, ids=edge_targets
            )  # Shape [E, H]

            num_incoming_to_node_per_message = tf.nn.embedding_lookup(
                params=type_to_num_incoming_edges[edge_type_idx, :], ids=edge_targets
            )  # Shape [E, H]

            # Calculate the messages themselves:
            messages = self._message_function(
                edge_source_states,
                edge_target_states,
                num_incoming_to_node_per_message,
                edge_type_idx,
                training,
            )

            messages_per_type.append(messages)
        return messages_per_type


MESSAGE_PASSING_IMPLEMENTATIONS: Dict[str, MessagePassing] = {}


def register_message_passing_implementation(cls):
    """Decorator used to register a message passing class implementation"""
    MESSAGE_PASSING_IMPLEMENTATIONS[cls.__name__.lower()] = cls
    return cls


def calculate_type_to_num_incoming_edges(node_embeddings, adjacency_lists):
    """Calculate the type_to_num_incoming_edges tensor.

        Returns:
            float32 tensor of shape [L, V] representing the number of incoming edges of
            a given type. Concretely, type_to_num_incoming_edges[l, v] is the number of
            edge of type l connecting to node v.

    >>> node_embeddings = tf.random.normal(shape=(5, 3))
    >>> adjacency_lists = [
    ...    tf.constant([[0, 1], [2, 4], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[2, 3], [2, 4]], dtype=tf.int32),
    ...    tf.constant([[3, 1]], dtype=tf.int32),
    ... ]
    ...
    >>> print(calculate_type_to_num_incoming_edges(node_embeddings, adjacency_lists))
    tf.Tensor(
    [[0. 1. 0. 0. 2.]
     [0. 0. 0. 1. 1.]
     [0. 1. 0. 0. 0.]], shape=(3, 5), dtype=float32)
    """

    type_to_num_incoming_edges = []
    for edge_type_adjacency_list in adjacency_lists:
        targets = edge_type_adjacency_list[:, 1]
        indices = tf.expand_dims(targets, axis=-1)
        num_incoming_edges = tf.scatter_nd(
            indices=indices,
            updates=tf.ones_like(targets, dtype=tf.float32),
            shape=(tf.shape(node_embeddings)[0],),
        )
        type_to_num_incoming_edges.append(num_incoming_edges)

    return tf.stack(type_to_num_incoming_edges)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
