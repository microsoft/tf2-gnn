"""Message passing layer."""
from abc import abstractmethod
from typing import Dict, List, NamedTuple, Tuple, Any
from dpu_utils.tf2utils import MLP
import tensorflow as tf


from tf2_gnn.utils.param_helpers import (
    get_activation_function,
    get_aggregation_function,
)

def binary_round_positive_case(num, digits):
    x = num
    shifted = digits
    while x > 1.0:
        shifted = shifted - 1
        x = x / 2.0
    while x < 0.5:
        shifted = shifted + 1
        x = x * 2.0
    r = tf.round(x * (1 << digits))
    while shifted > 0:
        shifted = shifted - 1
        r = r / 2.0
    while shifted < 0:
        shifted = shifted + 1
        r = r * 2.0
    return r
def binary_round(num, digits=10):
    if num == 0.0:
        return 0.0
    elif num < 0.0:
        return -binary_round_positive_case(-num, digits)
    else:
        return binary_round_positive_case(num,digits)
def row_dealer(row):
    return tf.map_fn(binary_round,row)
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
            "num_edge_MLP_hidden_layers": 1,
        }

    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._hidden_dim = int(params["hidden_dim"])
        self.GPU=params["GPU"]

        aggregation_fn_name = params["aggregation_function"]
        self._aggregation_fn = get_aggregation_function(aggregation_fn_name)

        self._message_activation_before_aggregation = params.get(
            "message_activation_before_aggregation", False
        )

        activation_fn_name = params["message_activation_function"]
        self._activation_fn = get_activation_function(activation_fn_name)

        self._hyperedge_type_mlps: List[tf.keras.layers.Layer, ...] = []
        self._num_edge_MLP_hidden_layers = params["num_edge_MLP_hidden_layers"]

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

    def build(self, input_shapes: MessagePassingInput):
        node_embedding_shapes = input_shapes.node_embeddings
        adjacency_list = input_shapes.adjacency_lists
        num_edge_types = len(adjacency_list)

        for i, adjacency_list_for_edge_type in enumerate(adjacency_list):
            edge_arity = adjacency_list_for_edge_type[1]
            # print("edge_arity", adjacency_list_for_edge_type)
            edge_layer_input_size = tf.TensorShape((None, edge_arity * node_embedding_shapes[-1]))
            with tf.name_scope(f"edge_type_{i}"):
                for endpoint_idx in range(edge_arity):
                    with tf.name_scope(f"edge_arity_{endpoint_idx}"):
                        mlp = MLP(out_size=self._hidden_dim, hidden_layers=self._num_edge_MLP_hidden_layers,activation_fun=tf.nn.relu)
                        #mlp=tf.keras.layers.Dense(units=self._hidden_dim,use_bias=True,activation=tf.nn.relu,)
                        self._hyperedge_type_mlps.append(mlp)
                        self._hyperedge_type_mlps[-1].build(edge_layer_input_size)

        super().build(input_shapes)

    def call(self, inputs: MessagePassingInput, training: bool = False):
        """Call the message passing layer.

        Args:
            inputs: A tuple containing two items:
                node_embeddings: float32 tensor of shape [V, D], the original representation of each
                    node in the graph.
                adjacency_lists: Tuple of L adjacency lists, represented as int32 tensors of shape
                    [E, n]. Concretely, adjacency_lists[l][k,:] == [v1,v2,...,vn] means that the k-th edge
                    of type l connects node v1 to node vn.
            training: A bool that denotes whether we are in training mode.

        Returns:
            float32 tensor of shape [V, hidden_dim]
        """
        node_embeddings, adjacency_lists = (
            inputs.node_embeddings,
            inputs.adjacency_lists,
        )
        num_nodes = tf.shape(node_embeddings)[0]

        # Compute messages and message targets for each edge type:
        messages = []  # list of tensors of messages of shape [E, H]
        messages_targets = []  # list of indices indicating the node receiving the message
        counter = 0
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_arity = adjacency_list_for_edge_type.shape[1]
            # Compute edge embeddings update function inputs by concatening all connected nodes:
            edges_node_representations = []
            for endpoint_idx in range(edge_arity):
                node_idxs = adjacency_list_for_edge_type[:, endpoint_idx]
                edges_node_representations.append(tf.gather(params=node_embeddings, indices=node_idxs))
            raw_edge_representations = tf.concat(edges_node_representations, axis=-1)
            # Now actually compute one result per involved node, using a separate function for
            # each hyperedge endpoint:
            for endpoint_idx in range(edge_arity):
                target_state_ids = adjacency_list_for_edge_type[:, endpoint_idx]
                messages.append(
                    self._hyperedge_type_mlps[counter](raw_edge_representations, training)
                    #self._hyperedge_type_mlps[counter](raw_edge_representations) #if use Dense layer
                )
                messages_targets.append(target_state_ids)
                counter = counter + 1

        messages = tf.concat(messages, axis=0)  # Shape [M, H]
        messages_targets = tf.concat(messages_targets, axis=0)  # Shape [M]
        # Node embedding
        aggregated_messages = tf.math.unsorted_segment_sum(
            data=messages,
            segment_ids=messages_targets,
            num_segments=num_nodes
        )
        #tf.test.is_gpu_available()
        aggregated_messages =  (lambda : self._my_tf_round(aggregated_messages,2) if self.GPU==True  else aggregated_messages)()
        #tf.print("before",aggregated_messages)
        #aggregated_messages = (lambda: tf.map_fn(row_dealer,aggregated_messages) if self.GPU == True else aggregated_messages)()
        #tf.print("after",aggregated_messages)
        return tf.nn.relu(aggregated_messages)


        # messages_per_type = self._calculate_messages_per_type(
        #     adjacency_lists, node_embeddings, training
        # )
        #
        # edge_type_to_message_targets = [
        #     adjacency_list_for_edge_type[:, 1]
        #     for adjacency_list_for_edge_type in adjacency_lists
        # ]
        #
        # new_node_states = self._compute_new_node_embeddings(
        #     node_embeddings,
        #     messages_per_type,
        #     edge_type_to_message_targets,
        #     num_nodes,
        #     training,
        # )  # Shape [V, H]
        #
        # return new_node_states

    def _my_tf_round(self,x, decimals=0): #trauncate
        multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
        return tf.cast(tf.cast(tf.round(x * multiplier),tf.int32),tf.float32) / multiplier

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
