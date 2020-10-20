from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel
from tf2_gnn import GNNInput, GNN

class InvariantArgumentSelectionTask(GraphTaskModel):
    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
        self._params = params
        self._num_edge_types = dataset.num_edge_types
        self._embedding_layer = tf.keras.layers.Embedding(
            input_dim=params["node_vocab_size"], #size of the vocabulary
            output_dim=params["node_label_embedding_size"]
        )
        self._gnn = GNN(params) #RGCN,RGIN,RGAT,GGNN
        self._regression_layers=[]
        for mlp_node in self._params["regression_hidden_layer_size"]:
            self._regression_layers.append(tf.keras.layers.Dense(
            units=mlp_node, activation=tf.nn.relu, use_bias=True))


        self._node_repr_output_layer = tf.keras.layers.Dense(units=1, use_bias=True)#we didn't normalize label so this should not be sigmoid
        self._node_to_graph_aggregation = None

    def build(self, input_shapes):
        # print("--build--")
        # build node embedding layer
        with tf.name_scope("Node_embedding_layer"):
            self._embedding_layer.build(tf.TensorShape((None,)))
        # build gnn layers
        self._gnn.build(
            GNNInput(
                node_features=tf.TensorShape((None, self._params["node_label_embedding_size"])),
                adjacency_lists=tuple(
                    input_shapes[f"adjacency_list_{edge_type_idx}"]
                    for edge_type_idx in range(self._num_edge_types)
                ),
                node_to_graph_map=tf.TensorShape((None,)),
                num_graphs=tf.TensorShape(()),
            )
        )

        #build task-specific layer
        with tf.name_scope("node_repr_to_regression_layer"):
            self._regression_layers[0].build(tf.TensorShape((None, self._params["hidden_dim"]))) #decide layer input shape

        for i,regression_layer in enumerate(self._regression_layers):
            if i>0:
                with tf.name_scope("regression_layer_"+str(i)):
                    regression_layer.build(tf.TensorShape((None, self._params["regression_hidden_layer_size"][i-1])))
        # with tf.name_scope("regression_layer_1"):
        #     self._regression_layer_1.build(tf.TensorShape((None, self._params["regression_hidden_layer_size"][0])))
        with tf.name_scope("last_regression_layer"):
            self._node_repr_output_layer.build(
                tf.TensorShape((None, self._params["regression_hidden_layer_size"][-1])) #decide layer input shape
            )

        super().build_horn_graph()#by pass graph_task_mode (GraphTaskModel)' build because it will build another gnn layer
        #tf.keras.Model.build([])

    def call(self, inputs, training: bool = False):
        node_labels_embedded = self._embedding_layer(inputs["node_features"], training=training)

        adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            inputs[f"adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )
        #before feed into gnn
        # print("node_features",inputs["node_features"])
        # print("node_features len",len(set(np.array(inputs["node_features"]))))
        # print("arguments",inputs["node_argument"])
        # print("node_to_graph_map",inputs['node_to_graph_map'])
        # print("num_graphs_in_batch",inputs['num_graphs_in_batch'])
        # print("adjacency_lists",adjacency_lists)

        # call gnn and get graph representation
        gnn_input = GNNInput(
            node_features=node_labels_embedded,
            num_graphs=inputs['num_graphs_in_batch'],
            node_to_graph_map=inputs['node_to_graph_map'],
            adjacency_lists=adjacency_lists
        )
        final_node_representations = self._gnn(gnn_input, training=training)
        node_representations=tf.gather(params=final_node_representations*1,indices=inputs["label_node_indices"])
        #print("argument_representations",argument_representations)
        return self.compute_task_output(inputs, node_representations, training)

    def compute_task_output(
            self,
            batch_features: Dict[str, tf.Tensor],
            final_node_representations: tf.Tensor,
            training: bool,
    ) -> Any:
        #call task specific layers
        #argument_regression_hidden_layer_output=self._node_repr_to_regression_layer(final_node_representations)
        x=self._regression_layers[0](final_node_representations)
        for i in range(1,len(self._regression_layers)):
            x=self._regression_layers[i](x)
        predicted_node_label = self._node_repr_output_layer(x)  # Shape [argument number, 1]
        return tf.squeeze(predicted_node_label, axis=-1) #Shape [predicted_node_label number,]

    def compute_task_metrics(#todo:change to hinge loss or lasso
            self,
            batch_features: Dict[str, tf.Tensor],
            task_output: Any,
            batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        mse = tf.losses.mean_squared_error(batch_labels["node_labels"], task_output)
        hinge_loss=tf.losses.hinge(batch_labels["node_labels"], task_output)
        mae = tf.losses.mean_absolute_error(batch_labels["node_labels"], task_output)
        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        return {
            "loss": mse,
            "batch_squared_error": mse * num_graphs,
            "batch_absolute_error": mae * num_graphs,
            "num_graphs": num_graphs,
        }

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_graphs = sum(
            batch_task_result["num_graphs"] for batch_task_result in task_results
        )
        total_absolute_error = sum(
            batch_task_result["batch_absolute_error"] for batch_task_result in task_results
        )
        epoch_mae = total_absolute_error / total_num_graphs
        return epoch_mae.numpy(), f"Mean Absolute Error = {epoch_mae.numpy():.3f}"

class InvariantNodeIdentifyTask(GraphTaskModel):

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
        self._params = params
        self._num_edge_types = dataset.num_edge_types
        self._embedding_layer = tf.keras.layers.Embedding(
            input_dim=params["node_vocab_size"], #size of the vocabulary
            output_dim=params["node_label_embedding_size"]
        )
        self._gnn = GNN(params) #RGCN,RGIN,RGAT,GGNN
        self._regression_layers = []
        for mlp_node in self._params["regression_hidden_layer_size"]:
            self._regression_layers.append(tf.keras.layers.Dense(
                units=mlp_node, activation=tf.nn.relu, use_bias=True))
        self._node_repr_output_layer = tf.keras.layers.Dense(activation=tf.nn.sigmoid,
            units=1, use_bias=True)  # we didn't normalize label so this should not be sigmoid
        self._node_to_graph_aggregation = None

    def build(self, input_shapes):
        # print("--build--")
        # build node embedding layer
        with tf.name_scope("Node_embedding_layer"):
            self._embedding_layer.build(tf.TensorShape((None,)))
        # build gnn layers
        self._gnn.build(
            GNNInput(
                node_features=tf.TensorShape((None, self._params["node_label_embedding_size"])),
                adjacency_lists=tuple(
                    input_shapes[f"adjacency_list_{edge_type_idx}"]
                    for edge_type_idx in range(self._num_edge_types)
                ),
                node_to_graph_map=tf.TensorShape((None,)),
                num_graphs=tf.TensorShape(()),
            )
        )

        #build task-specific layer
        with tf.name_scope("node_repr_to_regression_layer"):
            self._regression_layers[0].build(
                tf.TensorShape((None, self._params["hidden_dim"])))  # decide layer input shape

        for i, regression_layer in enumerate(self._regression_layers):
            if i > 0:
                with tf.name_scope("regression_layer_" + str(i)):
                    regression_layer.build(tf.TensorShape((None, self._params["regression_hidden_layer_size"][i - 1])))
        # with tf.name_scope("regression_layer_1"):
        #     self._regression_layer_1.build(tf.TensorShape((None, self._params["regression_hidden_layer_size"][0])))
        with tf.name_scope("last_regression_layer"):
            self._node_repr_output_layer.build(
                tf.TensorShape((None, self._params["regression_hidden_layer_size"][-1]))  # decide layer input shape
            )
        super().build_horn_graph()#by pass graph_task_mode (GraphTaskModel)' build because it will build another gnn layer
        #tf.keras.Model.build([])

    def call(self, inputs, training: bool = False):
        node_labels_embedded = self._embedding_layer(inputs["node_features"], training=training)

        adjacency_lists: Tuple[tf.Tensor, ...] = tuple(
            inputs[f"adjacency_list_{edge_type_idx}"]
            for edge_type_idx in range(self._num_edge_types)
        )

        # call gnn and get graph representation
        gnn_input = GNNInput(
            node_features=node_labels_embedded,
            num_graphs=inputs['num_graphs_in_batch'],
            node_to_graph_map=inputs['node_to_graph_map'],
            adjacency_lists=adjacency_lists
        )
        final_node_representations = self._gnn(gnn_input, training=training)
        if self._params["label_type"]=="argument_identify":
            return self.compute_task_output(inputs, final_node_representations, training)
        elif self._params["label_type"] == "control_location_identify":
            return self.compute_task_output(inputs, final_node_representations, training)
        elif self._params["label_type"] == "predicate_occurrence_in_SCG":
            final_node_representations = self._gnn(gnn_input, training=training)
            node_representations = tf.gather(params=final_node_representations * 1,
                                             indices=inputs["label_node_indices"])
            return self.compute_task_output(inputs, node_representations, training)
        elif self._params["label_type"]=="argument_identify_no_batchs":
            current_node_representations = tf.gather(params=final_node_representations * 1,
                                                     indices=inputs["current_node_index"])
            return self.compute_task_output(inputs, current_node_representations, training)


    def compute_task_output(
        self,
        batch_features: Dict[str, tf.Tensor],
        final_node_representations: tf.Tensor,
        training: bool,
    ) -> Any:
        #call task specific layers
        x = self._regression_layers[0](final_node_representations)
        for i in range(1, len(self._regression_layers)):
            x = self._regression_layers[i](x)
        predicted_node_label = self._node_repr_output_layer(x)  # Shape [argument number, 1]

        return tf.squeeze(predicted_node_label, axis=-1)

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        ce = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true=batch_labels["node_labels"], y_pred=task_output, from_logits=False
            )
        )
        num_correct = tf.reduce_sum(
            tf.cast(
                tf.math.equal(batch_labels["node_labels"], tf.math.round(task_output)), tf.int32
            )
        )
        num_nodes = tf.cast(len(batch_labels["node_labels"]), tf.float32)
        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        return {
            "loss": ce,
            "batch_acc": tf.cast(num_correct, tf.float32) / num_nodes,
            "num_correct": num_correct,
            "num_graphs": num_graphs,
            "num_nodes":num_nodes
        }

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        total_num_graphs = np.sum(
            batch_task_result["num_graphs"] for batch_task_result in task_results
        )
        total_num_nodes = np.sum(
            batch_task_result["num_nodes"] for batch_task_result in task_results
        )
        total_num_correct = np.sum(
            batch_task_result["num_correct"] for batch_task_result in task_results
        )
        epoch_acc = tf.cast(total_num_correct, tf.float32) / total_num_nodes
        return -epoch_acc.numpy(), f"Accuracy = {epoch_acc.numpy():.3f}"




