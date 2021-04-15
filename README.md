# Graph Neural Networks in TF2

Implementation and example training scripts of various flavours of graph neural network in 
TensorFlow 2.0.
Much of it is based on the code in the [tf-gnn-samples](https://github.com/microsoft/tf-gnn-samples) repo.

## Installation
You can install the `tf2_gnn` module from the Python Package Index using 
`pip install tf2_gnn`.

Alternatively (for example, for development), you can check out this repository,
navigate to it and run `pip install -e ./` to install it as a local editable package.

You will then be able to use the `tf2_gnn.layers.GNN` class and related utilities.

This code was tested in Python 3.6 and 3.7 with TensorFlow 2.0 and 2.1.

The code is maintained by the [All Data AI](https://www.microsoft.com/en-us/research/group/ada/)
group at Microsoft Research, Cambridge, UK.
We are [hiring](https://www.microsoft.com/en-us/research/theme/ada/#!opportunities).


## Testing the Installation

To test if all components are set up correctly, you can run a simple experiment on the
protein-protein interaction (PPI) task first described by 
[Zitnik & Leskovec, 2017](#zitnik-leskovec-2017).
You can download the data for this task from https://data.dgl.ai/dataset/ppi.zip
and unzip it into a local directory (e.g., `data/ppi`).
Then, you can use the convenience utility `tf2_gnn_train` (see `--help` for a description
of options) to train a Relational Graph Convoluational Network model as follows:
```
$ tf2_gnn_train RGCN PPI --max-epochs 10 data/ppi/
Setting random seed 0.
Trying to load task/model-specific default parameters from /dpuhome/files/users/mabrocks/Projects/TF2-GNN/tf2_gnn/cli_utils/default_hypers/PPI_RGCN.json ... File found.
 Dataset default parameters: {'max_nodes_per_batch': 10000, 'add_self_loop_edges': True, 'tie_fwd_bkwd_edges': False}
Loading data from data/ppi/.
 Loading PPI train data from data/ppi/.
 Loading PPI valid data from data/ppi/.
[...]
Dataset parameters: {"max_nodes_per_batch": 8000, "add_self_loop_edges": true, "tie_fwd_bkwd_edges": false}
Model parameters: {"gnn_aggregation_function": "sum", "gnn_message_activation_function": "ReLU", "gnn_hidden_dim": 320, "gnn_use_target_state_as_input": false, "gnn_normalize_by_num_incoming": true, "gnn_num_edge_MLP_hidden_layers": 0, "gnn_message_calculation_class": "RGCN", "gnn_initial_node_representation_activation": "tanh", "gnn_dense_intermediate_layer_activation": "tanh", "gnn_num_layers": 4, "gnn_dense_every_num_layers": 10000, "gnn_residual_every_num_layers": 10000, "gnn_use_inter_layer_layernorm": false, "gnn_layer_input_dropout_rate": 0.1, "gnn_global_exchange_mode": "gru", "gnn_global_exchange_every_num_layers": 10000, "gnn_global_exchange_weighting_fun": "softmax", "gnn_global_exchange_num_heads": 4, "gnn_global_exchange_dropout_rate": 0.2, "optimizer": "Adam", "learning_rate": 0.001, "learning_rate_decay": 0.98, "momentum": 0.85, "gradient_clip_value": 1.0}
Initial valid metric: Avg MicroF1: 0.368.
   (Stored model metadata to trained_model/RGCN_PPI__2020-02-25_11-10-38_best.pkl and weights to trained_model/RGCN_PPI__2020-02-25_11-10-38_best.hdf5)
== Epoch 1
 Train:  25.6870 loss | Avg MicroF1: 0.401 | 2.63 graphs/s
 Valid:  33.1668 loss | Avg MicroF1: 0.419 | 4.01 graphs/s
  (Best epoch so far, target metric decreased to -0.41886 from -0.36762.)
   (Stored model metadata to trained_model/RGCN_PPI__2020-02-25_11-10-38_best.pkl and weights to trained_model/RGCN_PPI__2020-02-25_11-10-38_best.hdf5)
[...]
```

After training finished, `tf2_gnn_test trained_model/RGCN_PPI__2020-02-25_11-10-38_best.pkl data/ppi` can be used to test the trained model.


# Code Structure

## Layers

The core functionality of the library is implemented as TensorFlow 2 (Keras) layers,
enabling easy integration into other code.


### `tf2_gnn.layers.GNN`

This implements a deep Graph Neural Network, stacking several layers of message passing.
On construction, a dictionary of hyperparameters needs to be provided (default
values can be obtained from `GNN.get_default_hyperparameters()`).
These hyperparameters configure the exact stack of GNN layers:
* `"num_layers"` sets the number of GNN message passing layers (usually, a number
  between 2 and 16)

* `"message_calculation_class"` configures the message passing style.
  This chooses the `tf2_gnn.layers.message_passing.*` layer used in each step.
  
  We currently support the following:
    * `GGNN`: Gated Graph Neural Networks ([Li et al., 2015](#li-et-al-2015)).
    * `RGCN`: Relational Graph Convolutional Networks ([Schlichtkrull et al., 2017](#schlichtkrull-et-al-2017)).
    * `RGAT`: Relational Graph Attention Networks ([Veličković et al., 2018](#veličković-et-al-2018)).
    * `RGIN`: Relational Graph Isomorphism Networks ([Xu et al., 2019](#xu-et-al-2019)).
    * `GNN-Edge-MLP`: Graph Neural Network with Edge MLPs - a variant of RGCN in which messages on edges are computed using full MLPs, not just a single layer applied to the source state.
    * `GNN-FiLM`: Graph Neural Networks with Feature-wise Linear Modulation ([Brockschmidt, 2019](#brockschmidt-2019)) - a new extension of RGCN with FiLM layers.
  
  Some of these expose additional hyperparameters; refer to their implementation for
  details.

* `"hidden_dim"` sets the size of the output of all message passing layers.

* `"layer_input_dropout_rate"` sets the dropout rate (during training) for the
  input of each message passing layer.

* `"residual_every_num_layers"` sets how often a residual connection is inserted
  between message passing layers. Concretely, a value of `k` means that every layer
  `l` that is a multiple of `k` (and only those!) will not receive the outputs of
  layer `l-1` as input, but instead the mean of the outputs of layers `l-1` and `l-k`.

* `"use_inter_layer_layernorm"` is a boolean flag indicating if `LayerNorm` should be
  used between different message passing layers.

* `"dense_every_num_layers"` configures how often a per-node representation dense layer
  is inserted between the message passing layers.
  Setting this to a large value (greather than `"num_layers"`) means that no dense
  layers are inserted at all.
  
  `"dense_intermediate_layer_activation"` configures the activation function used after
  the dense layer; the default of `"tanh"` can help stabilise training of very deep
  GNNs.

* `"global_exchange_every_num_layers"` configures how often a graph-level exchange of
  information is performed.
  For this, a graph level representation (see `tf2_gnn.layers.NodesToGraphRepresentation`
  below) is computed and then used to update the representation of each node.
  The style of this update is configured by `"global_exchange_mode"`, offering three
  modes:
    * `"mean"`, which just computes the arithmetic mean of the node and graph-level
      representation.
    * `"mlp"`, which computes a new representation using an MLP that gets the
      concatenation of node and graph level representations as input.
    * `"gru"`, which uses a GRU cell that gets the old node representation as state
      and the graph representation as input.

The `GNN` layer takes a `GNNInput` named tuple as input, which encapsulates initial
node features, adjacency lists, and auxiliary information.
The easiest way to construct such a tuple is to use the provided [dataset](datasets)
classes in combination with the provided [model](models).


### `tf2_gnn.layers.NodesToGraphRepresentation`

This implements the task of computing a graph-level representation given node-level
representations (e.g., obtained by the `GNN` layer).

Currently, this is only implemented by the `WeightedSumGraphRepresentation` layer,
which produces a graph representation by a multi-headed weighted sum of (transformed) 
node representations, configured by the following hyperparameters set in the
layer constructor:
* `graph_representation_size` sets the size of the computed representation.
  By setting this to `1`, this layer can be used to directly implement graph-level
  regression tasks.
* `num_heads` configures the number of parallel (independent) weighted sums that
  are computed, whose results are concatenated to obtain the final result.
  Note that this means that the `graph_representation_size` needs to be a multiple
  of the `num_heads` value.
* `weighting_fun` can take two values:
  * `"sigmoid"` computes a weight for each node independently by first computing
    a per-node score, which is then squashed through a sigmoid.
    This is appropriate for tasks that are related to counting occurrences of a 
    feature in a graph, where the node weight is used to ignore certain nodes.
  * `"softmax"` computes weights for all graph nodes together by first computing
    per-node scores, and then performing a softmax over all scores.
    This is appropriate for tasks that require identifying important parts of
    the graph.
* `scoring_mlp_layers`, `scoring_mlp_activation_fun`, `scoring_mlp_dropout_rate`
  configure the MLP that computes the per-node scores.
* `transformation_mlp_layers`, `transformation_mlp_activation_fun`, 
  `transformation_mlp_dropout_rate` configure the MLP that computes the
  transformed node representations that are summed up.


## Datasets

We use a sparse representation of graphs, which requires a complex batching strategy
in which the graphs making up a minibatch are joined into a single graph of many
disconnected components.
The extensible `tf2_gnn.data.GraphDataset` class implements this procedure, and can
be subclassed to handle task-specific datasets and additional properties.
It exposes a `get_tensorflow_dataset` method that can be used to obtain a 
`tf.data.Dataset` that can be used in training/evaluation loops.

We currently provide three implementations of this:
* `tf2_gnn.data.PPIDataset` implements reading the protein-protein interaction (PPI)
  data first used by [Zitnik & Leskovec, 2017](#zitnik-leskovec-2017).
* `tf2_gnn.data.QM9Dataset` implements reading the quantum chemistry data first
  used by [Ramakrishnan et al., 2014](#ramakrishnan-et-al-2014).
* `tf2_gnn.data.JsonLGraphPropertyDataset` implements reading a generic dataset
  made up of graphs with a single property, stored in JSONLines format:
  * Files "train.jsonl.gz", "valid.jsonl.gz" and "test.jsonl.gz" are expected to
    store the train/valid/test datasets.
  * Each of the files is gzipped text file in which each line is a valid
    JSON dictionary with
    * a `"graph"` key, which in turn points to a dictionary with keys
      * `"node_features"` (list of numerical initial node labels),
      * `"adjacency_lists"` (list of list of directed edge pairs),
    * a `"Property"` key having a a single floating point value.


## Models

We provide some built-in models in `tf2_gnn.models`, which can either be directly
re-used or serve as inspiration for other models:
* `tf2_gnn.models.GraphRegressionTask` implements a graph-level regression model,
  for example to make molecule-level predictions such as in the QM9 task.
* `tf2_gnn.models.GraphBinaryClassificationTask` implements a binary classification
  model.
* `tf2_gnn.models.NodeMulticlassTask` implements a node-level multiclass classification
  model, suitable to implement the PPI task.


## Tasks

Tasks are a combination of datasets, models and specific hyperparameter settings.
These can be registered (and then used by name) using the utilities in
`tf2_gnn.utils.task_utils` (where a few default tasks are defined as well) and then
used in tools such as `tf2_gnn_train`.

# Authors

* [Henry Jackson-Flux](mailto:Henry.JacksonFlux@microsoft.com)
* [Marc Brockschmidt](mailto:Marc.Brockschmidt@microsoft.com)
* [Megan Stanley](mailto:t-mestan@microsoft.com)
* [Pashmina Cameron](mailto:Pashmina.Cameron@microsoft.com)


# References

#### Brockschmidt, 2019
Marc Brockschmidt. GNN-FiLM: Graph Neural Networks with Feature-wise Linear
Modulation. (https://arxiv.org/abs/1906.12192)

#### Li et al., 2015
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated Graph
Sequence Neural Networks. In International Conference on Learning
Representations (ICLR), 2016. (https://arxiv.org/pdf/1511.05493.pdf)

#### Ramakrishnan et al., 2014
Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole
Von Lilienfeld. Quantum Chemistry Structures and Properties of 134 Kilo
Molecules. Scientific Data, 1, 2014.
(https://www.nature.com/articles/sdata201422/)

#### Schlichtkrull et al., 2017
Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg,
Ivan Titov, and Max Welling. Modeling Relational Data with Graph
Convolutional Networks. In Extended Semantic Web Conference (ESWC), 2018.
(https://arxiv.org/pdf/1703.06103.pdf)

#### Veličković et al. 2018
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Liò, and Yoshua Bengio. Graph Attention Networks. In International Conference
on Learning Representations (ICLR), 2018. (https://arxiv.org/pdf/1710.10903.pdf)

#### Xu et al. 2019
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are
Graph Neural Networks? In International Conference on Learning Representations
(ICLR), 2019. (https://arxiv.org/pdf/1810.00826.pdf)

#### Zitnik & Leskovec, 2017
Marinka Zitnik and Jure Leskovec. Predicting Multicellular Function Through
Multi-layer Tissue Networks. Bioinformatics, 33, 2017.
(https://arxiv.org/abs/1707.04638)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
