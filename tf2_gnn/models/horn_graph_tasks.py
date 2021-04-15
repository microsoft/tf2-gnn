from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel
from tf2_gnn import GNNInput, GNN
import math

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
            #todo: add regularizers?
            self._regression_layers.append(tf.keras.layers.Dense(
            units=mlp_node, activation=tf.nn.relu, use_bias=True))

        if self._params["label_type"] == "argument_bound":
            self._node_repr_output_layer = tf.keras.layers.Dense(units=2, use_bias=True)
        else:
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
        #print("self._gnn.weights",self._gnn.weights)
        #print("final_node_representations",final_node_representations)
        node_representations=tf.gather(params=final_node_representations*1,indices=inputs["label_node_indices"])
        #print("node_representations",node_representations)
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
        if self._params["label_type"] == "argument_bound":
            return predicted_node_label
        else:
            return tf.squeeze(predicted_node_label, axis=-1) #Shape [predicted_node_label number,]

    def compute_task_metrics(
            self,
            batch_features: Dict[str, tf.Tensor],
            task_output: Any,
            batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        mse = tf.losses.mean_squared_error(batch_labels["node_labels"], task_output)
        if math.isnan(mse):
            print("node_labels",batch_labels["node_labels"])
            print("task_output",task_output)
        #hinge_loss=tf.losses.hinge(batch_labels["node_labels"], task_output)
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
        if self._params["label_type"] == "argument_bound":
            #return epoch_mae.numpy(), "Mean Absolute Error = "+str(epoch_mae.numpy())
            return epoch_mae.numpy(), f"Mean Absolute Error = {epoch_mae.numpy()}"
        else:
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
        self._node_repr_output_layer = tf.keras.layers.Dense(activation=tf.nn.sigmoid,units=1, use_bias=True)  # sigmoid?
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
        if self._params["label_type"]=="argument_identify" or self._params["label_type"]=="control_location_identify":
            return self.compute_task_output(inputs, final_node_representations, training)
        elif self._params["label_type"]=="argument_identify_no_batchs":
            current_node_representations = tf.gather(params=final_node_representations * 1,
                                                     indices=inputs["current_node_index"])
            return self.compute_task_output(inputs, current_node_representations, training)
        elif self._params["label_type"] in self._params["gathered_nodes_binary_classification_task"]:
            node_representations = tf.gather(params=final_node_representations * 1,
                                             indices=inputs["label_node_indices"])
            return self.compute_task_output(inputs, node_representations, training)


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
        # binary_crossentropy_value=tf.keras.losses.binary_crossentropy(y_true=batch_labels["node_labels"], y_pred=task_output,from_logits=True)
        # ce = tf.reduce_mean(binary_crossentropy_value)
        # description: set class weight here
        class_weight=self._params["class_weight"]
        class_weighted_binary_crossentropy_value = tf.nn.weighted_cross_entropy_with_logits(batch_labels["node_labels"],task_output,class_weight["weight_for_1"]/class_weight["weight_for_0"])
        #self.get_weighted_binary_crossentropy(batch_labels["node_labels"], task_output, class_weight,from_logits=True)
        ce = tf.reduce_mean(class_weighted_binary_crossentropy_value)

        if math.isnan(ce):
            print("batch_features",len(batch_features))
            print("batch_features", batch_features)
            print("label_node_indices",batch_features["label_node_indices"])
            print("labels", batch_labels["node_labels"])
            print("task_output", task_output)
            print("loss ce", ce)
        #description: set round threshold here
        num_correct = tf.reduce_sum(
            tf.cast(
                tf.math.equal(batch_labels["node_labels"], self.my_round_fun(task_output,threshold=self._params["threshold"])), tf.int32
            )
        )
        # num_correct = tf.reduce_sum(
        #     tf.cast(
        #         tf.math.equal(batch_labels["node_labels"], tf.math.round(task_output)), tf.int32
        #     )
        # )
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

    def get_weighted_binary_crossentropy(self, true_y, predicted_y,class_weight={},from_logits=True):
        #ce = 0
        ce_raw=0
        for y,y_hat in zip(true_y,predicted_y):
            #y_hat = np.round((lambda: self.sigmoid(y_hat) if from_logits == True else self.logit(y_hat))(),2)
            # ce_raw = ce_raw + (-y * np.log((self.sigmoid(y_hat))) * class_weight["weight_for_1"]) \
            #          - ((1 - y) * np.log(1 - (self.sigmoid(y_hat))) * class_weight["weight_for_0"])
            # l = (1 + ((class_weight["weight_for_1"]/class_weight["weight_for_0"]) - 1) * y)
            # ce_raw = ce_raw + (1 - y) * y_hat + l * (np.log(1 + np.exp(-abs(y_hat))) + max(-y_hat, 0))
            ce_raw=ce_raw + tf.nn.weighted_cross_entropy_with_logits([y],[y_hat],class_weight["weight_for_1"]/class_weight["weight_for_0"])
            # if (y == 1):
            #     ce = ce + tf.keras.losses.binary_crossentropy([y], [y_hat],from_logits=from_logits) * class_weight["weight_for_1"]
            # if (y == 0):
            #     ce = ce + tf.keras.losses.binary_crossentropy([y], [y_hat],from_logits=from_logits) * class_weight["weight_for_0"]
        return ce_raw/len(true_y)
        #return ce / len(true_y)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def logit(self,p):
        return np.log(p/(1-p))

    def my_round_fun(nself,num_list, threshold):
        return [1 if num > threshold else 0 for num in num_list]






