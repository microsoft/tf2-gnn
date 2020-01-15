# Graph Neural Networks in TF2

Implementation and example training scripts of various flavours of graph neural network in 
TensorFlow 2.0.
Much of it is based on the code in the [tf-gnn-samples](https://github.com/microsoft/tf-gnn-samples) repo.

## Installation
To install the `tf2_gnn` module, navigate to this directory and run
```python
pip install ./
```
You will then be able to use the `tf2_gnn.GNN` class and related utilities.

This code was tested in Python 3.6 and 3.7 with TensorFlow 2.0 and 2.1.
To install required packages, run `pip install -r requirements.txt`.

The code is maintained by the [All Data AI](https://www.microsoft.com/en-us/research/group/ada/)
group at Microsoft Research, Cambridge, UK.
We are [hiring](https://www.microsoft.com/en-us/research/theme/ada/#!opportunities).

## Models
Currently, six model types are implemented:
* `GGNN`: Gated Graph Neural Networks ([Li et al., 2015](#li-et-al-2015)).
* `RGCN`: Relational Graph Convolutional Networks ([Schlichtkrull et al., 2017](#schlichtkrull-et-al-2017)).
* `RGAT`: Relational Graph Attention Networks ([Veličković et al., 2018](#veličković-et-al-2018)).
* `RGIN`: Relational Graph Isomorphism Networks ([Xu et al., 2019](#xu-et-al-2019)).
* `GNN-Edge-MLP`: Graph Neural Network with Edge MLPs - a variant of RGCN in which messages on edges are computed using full MLPs, not just a single layer applied to the source state.
* `GNN-FiLM`: Graph Neural Networks with Feature-wise Linear Modulation ([Brockschmidt, 2019](#brockschmidt-2019)) - a new extension of RGCN with FiLM layers.

## Tasks
Tasks are viewed as a specific combination of a dataset and a model. The
interface (and some examples) for this can be found in `tf2_gnn/utils/task_utils.py`.
Currently, four tasks are implemented, exposing different models:

### PPI
The `PPI` task (implemented in `tasks/ppi_dataset.py`) handles the protein-protein
interaction task first described by [Zitnik & Leskovec, 2017](#zitnik-leskovec-2017).
The implementation illustrates how to handle the case of inductive graph
learning with node-level predictions.
To run experiments on this task, you need to download the data from
https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip, unpack it (for
example into `data/ppi`) and can train a model by executing
`python train.py MODEL PPI data/ppi`.

### QM9
The `QM9` task (implemented in `tasks/qm9_dataset.py`) handles the quantum chemistry
prediction tasks first described by [Ramakrishnan et al., 2014](#ramakrishnan-et-al-2014)
The implementation illustrates how to handle the case of inductive graph
learning with graph-level predictions.
You can call this by running `python train.py MODEL QM9 data/qm9`.


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

This project welcomes contributions and suggestions.  Most contributions
require you to agree to a Contributor License Agreement (CLA) declaring 
that you have the right to, and actually do, grant us the rights to use
your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine 
whether you need to provide a CLA and decorate the PR appropriately (e.g.,
label, comment). Simply follow the instructions provided by the bot.
You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

