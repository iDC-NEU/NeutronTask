#!/bin/bash
###
 # @Description: 
 # @Author: lusz
 # @Date: 2024-12-13 15:08:45
### 

# 定义参数范围
edges=(80000000 160000000 320000000)
dims=(256 512 1024)
labels=(16 32 64)
train_ratios=(0.1 0.5 0.8)

# 定义中间值索引
mid_edges=1
mid_dims=1
mid_labels=1
mid_train_ratios=1

# 数据文件夹路径
data_dir="/home/hdd2/dataset/random"

# 保存结果的文件
result_file="epoch_times.txt"
echo "parameter,epoch_time" > $result_file

# 遍历 edges
for e in "${edges[@]}"; do
    echo "Running experiment with edges=$e"
    
    # 选择对应的 edge 文件
    if [ $e -eq 80000000 ]; then
        edge_file="${data_dir}/V_2M_E_80M.edge"
    elif [ $e -eq 160000000 ]; then
        edge_file="${data_dir}/V_2M_E_160M.edge"
    elif [ $e -eq 320000000 ]; then
        edge_file="${data_dir}/V_2M_E_320M.edge"
    else
        echo "Error: Unsupported edge count $e"
        exit 1
    fi

    # 运行 Python 脚本并将结果保存到文件中
    python node_classification_random.py --edge_file $edge_file --num_edges $e --feature_dim ${dims[$mid_dims]} --num_classes ${labels[$mid_labels]} --train_ratio ${train_ratios[$mid_train_ratios]} | tee -a $result_file
    echo "edges=$e" >> $result_file
done

# 遍历 dims
for d in "${dims[@]}"; do
    echo "Running experiment with dims=$d"
    python node_classification_random.py --edge_file ${edge_file} --num_edges ${edges[$mid_edges]} --feature_dim $d --num_classes ${labels[$mid_labels]} --train_ratio ${train_ratios[$mid_train_ratios]} | tee -a $result_file
    echo "dims=$d" >> $result_file
done

# 遍历 labels
for l in "${labels[@]}"; do
    echo "Running experiment with labels=$l"
    python node_classification_random.py --edge_file ${edge_file} --num_edges ${edges[$mid_edges]} --feature_dim ${dims[$mid_dims]} --num_classes $l --train_ratio ${train_ratios[$mid_train_ratios]} | tee -a $result_file
    echo "labels=$l" >> $result_file
done

# 遍历 train_ratios
for tr in "${train_ratios[@]}"; do
    echo "Running experiment with train_ratio=$tr"
    python node_classification_random.py --edge_file ${edge_file} --num_edges ${edges[$mid_edges]} --feature_dim ${dims[$mid_dims]} --num_classes ${labels[$mid_labels]} --train_ratio $tr | tee -a $result_file
    echo "train_ratio=$tr" >> $result_file
done

echo "Experiments completed. Results saved to $result_file."
