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

# 保存结果的文件
result_file="epoch_times.txt"
echo "parameter,epoch_time" > $result_file

# 遍历 edges
for e in "${edges[@]}"; do
    epoch_time=$(python node_classitication_random.py --num_edges $e --feature_dim ${dims[$mid_dims]} --num_classes ${labels[$mid_labels]} --train_ratio ${train_ratios[$mid_train_ratios]}  | grep "epoch_time" | awk '{print $2}')
    echo "edges=$e,$epoch_time" >> $result_file
done

# 遍历 dims
for d in "${dims[@]}"; do
    epoch_time=$(python node_classitication_random.py --num_edges ${edges[$mid_edges]} --feature_dim $d --num_classes ${labels[$mid_labels]} --train_ratio ${train_ratios[$mid_train_ratios]}  | grep "epoch_time" | awk '{print $2}')
    echo "dims=$d,$epoch_time" >> $result_file
done

# 遍历 labels
for l in "${labels[@]}"; do
    epoch_time=$(python node_classitication_random.py --num_edges ${edges[$mid_edges]} --feature_dim ${dims[$mid_dims]} --num_classes $l --train_ratio ${train_ratios[$mid_train_ratios]}  | grep "epoch_time" | awk '{print $2}')
    echo "labels=$l,$epoch_time" >> $result_file
done

# 遍历 train_ratios
for tr in "${train_ratios[@]}"; do
    epoch_time=$(python node_classitication_random.py --num_edges ${edges[$mid_edges]} --feature_dim ${dims[$mid_dims]} --num_classes ${labels[$mid_labels]} --train_ratio $tr  | grep "epoch_time" | awk '{print $2}')
    echo "train_ratio=$tr,$epoch_time" >> $result_file
done

echo "Experiments completed. Results saved to $result_file."
