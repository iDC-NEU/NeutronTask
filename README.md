<!--
 * @Author: fzb fzb0316@163.com
 * @Date: 2024-09-29 18:52:21
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-30 19:10:17
 * @FilePath: /fuzb/NtsTask/NeutronTask/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

**NeutronTask** is a Multi-GPU Graph Neural Networks (GNN) training framework that design GNN task parallelism and task-decoupled GNN training. 



The code is implemented based on [NeutronStar](https://github.com/iDC-NEU/NeutronStarLite), extending its single-GPU execution to a multi-GPU platform.



## Quick Start

A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

**cmake** >=3.16.3

**MPI** for inter-process communication 

**cuda** > 11.3 for GPU based graph operation.

**libnuma** for NUMA-aware memory allocation.

**cub** for GPU-based graph propagation


```
sudo apt install libnuma-dev"
```

**libtorch** version > 1.13 with gpu support for nn computation

unzip the **libtorch** package in the root dir of **NeutronStar** and change CMAKE_PREFIX_PATH in "CMakeList.txt"to your own path

download **cub** to the ./NeutronStar/cuda/ dictionary.


configure PATH and LD_LIBRARY_PATH for **cuda** and **mpi**
```
export CUDA_HOME=/usr/local/cuda-10.2
export MPI_HOME=/path/to/your/mpi
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$MPI_HOME/bin:$CUDA_HOME/bin:$PATH
```

**clang-format** is optional for auto-formatting: 
```shell
sudo apt install clang-format
```

please able **GPU compilation** with the following command at configure time:
```shell
cmake -DCUDA_ENABLE=ON ..
```

To build:
```shell
mkdir build

cd build

cmake ..

make -j4
```
single-machine multi-GPUs
```
./run_nts.sh 1 task_parallelism.cfg
```
## Dataset
### `.edge` File
- **Purpose**: Defines the graph's edges, representing the graph's topology.
- **Format**: Each line represents an edge between two nodes:

| node_id1 | node_id2 |
|----------|----------|
| 1        | 2        |
| 2        | 3        |
| 3        | 4        |
| 4        | 1        |
| 1        | 3        |

---

### `.feature` File
- **Purpose**: Stores features for each node in the graph.
- **Format**: Each row represents the feature vector of a node, optionally prefixed by the node ID:

| node_id | feature1 | feature2 | feature3 |
|---------|----------|----------|----------|
| 1       | 0.1      | 0.2      | 0.3      |
| 2       | 0.4      | 0.5      | 0.6      |
| 3       | 0.7      | 0.8      | 0.9      |
| 4       | 1.0      | 1.1      | 1.2      |

---

### `.label` File
- **Purpose**: Stores labels for nodes or edges for classification or regression tasks.
- **Format**: Each row represents a node/edge and its corresponding label:

| node_id | label |
|---------|-------|
| 1       | 0     |
| 2       | 1     |
| 3       | 0     |
| 4       | 1     |

---

### `.mask` File
- **Purpose**: Specifies which nodes or edges belong to training, validation, or testing sets.
- **Format**: Each row is a boolean or binary indicator (1/0) for a node's inclusion in a dataset:

| node_id | train_mask | val_mask | test_mask |
|---------|------------|----------|-----------|
| 1       | 1          | 0        | 0         |
| 2       | 0          | 1        | 0         |
| 3       | 0          | 0        | 1         |
| 4       | 1          | 0        | 0         |
## CFG file
| **Section**            | **Parameter**       | **Description**                                                      |
|------------------------|---------------------|----------------------------------------------------------------------|
| **General Parameters**  | ALGORITHM           | Specifies the algorithm to use (e.g., APPNP, GCN).                   |
|                        | Decoupled           | Set to 1 to enable decoupled training, 0 otherwise.                  |
|                        | SMALLGRAPH          | Set to 1 for small graph optimizations, 0 otherwise.                  |
| **Dataset Parameters**  | VERTICES            | Number of nodes in the graph.                                         |
|                        | LAYERS              | Defines the neural network architecture (e.g., input-hidden-output). |
|                        | EDGE_FILE           | Path to the graph's edge file.                                        |
|                        | FEATURE_FILE        | Path to the node feature file.                                        |
|                        | LABEL_FILE          | Path to the node label file.                                          |
|                        | MASK_FILE           | Path to the file with train/val/test masks.                           |
| **Training Parameters** | EPOCHS              | Number of training epochs.                                           |
|                        | LEARN_RATE          | Learning rate for the optimizer.                                      |
|                        | WEIGHT_DECAY        | Regularization to avoid overfitting.                                  |
|                        | DROP_RATE           | Dropout rate for regularization during training.                      |
| **Processing Parameters** | PROC_CUDA         | Set to 1 to use CUDA (GPU acceleration), 0 otherwise.                |
|                        | GPUNUM              | Number of GPUs to use.                                                |
|                        | PIPELINENUM         | Number of pipeline stages for processing.                             |
| **Algorithm-Specific Parameters** | ALPHA   | PageRank teleport probability for APPNP.                              |
|                        | K                   | Number of propagation iterations.                                     | 

## toolkits
| Toolkit       | Description                                                        |
|---------------|--------------------------------------------------------------------|
| **APPNP_DP**  | Approximate Personalized Propagation of Neural Predictions (APPNP) using Data Parallelism|
| **GCN_DP**    | Graph Convolutional Networks (GCN) with Data Parallelism using Data Parallelism         |
| **GCN_TP_TD_pipeline**    |  Graph Convolutional Networks (GCN) with Data Parallelism using Task Parallelism and Task Decoupled Training and pipeline|
| **GCN_TP_TD_pipeline_wopipeline** | Graph Convolutional Networks (GCN) with Data Parallelism using Task Parallelism and Task Decoupled Training                           |
| **GAT**   | Graph Attention Networks (GAT)          |
## baseline
### Sancus
bash baseline/sancus/light-dist-gnn/run.sh
### DGL
``` python 
bash baseline/dgl/examples/multigpu/run_random.sh
bash baseline/dgl/examples/pytorch/gcn/run.sh
```