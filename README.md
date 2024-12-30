<!--
 * @Author: fzb fzb0316@163.com
 * @Date: 2024-09-29 18:52:21
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-29 20:53:27
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

## Dataset

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

## toolkit介绍

## baseline跑法(脚本介绍)
### Sancus

single-machine multi-GPUs
```
./run_nts.sh 1 task_parallelism.cfg
```