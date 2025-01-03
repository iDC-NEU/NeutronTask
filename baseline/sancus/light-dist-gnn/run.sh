
###
 # @Author: fzb fzb0316@163.com
 # @Date: 2024-03-24 16:05:46
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-12-25 16:45:59
 # @FilePath: /light-dist-gnn/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 


# ogbn-products
# python main.py --dataset ogbn-products --nlayers 2 --hidden 128 --epoch 50  --model DecoupleGCN  --nprocs 2
# python main.py --dataset ogbn-products --nlayers 2 --hidden 128 --epoch 200  --model DecoupleGCN  --nprocs 1 --savepath ./outputs/products/products_gcn.txt
# python main.py --dataset ogbn-products --nlayers 2 --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/products/products_gcn.txt
# python main.py --dataset ogbn-products --nlayers 1 --hidden 128 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/products/products_gcn.txt
# cora
# python main.py --dataset cora --nlayers 2 --hidden 128 --epoch 50  --model DecoupleGCN  --nprocs 2
# python main.py --dataset cora --nlayers 2 --hidden 128 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/cora/cora_gcn.txt 
# python main.py --dataset cora  --nlayers 8 --hidden 128 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/cora/cora_gcn.txt 
# python main.py --dataset cora  --nlayers 2 --hidden 128 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/cora/cora_gcn.txt 

#reddit
# python main.py --dataset reddit --nlayers 2 --hidden 256 --epoch 50  --model DecoupleGCN  --nprocs 4
# python main.py --dataset reddit  --nlayers 2 --hidden 256 --epoch 10  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk4.txt 
# python main.py --dataset ogbn-products  --nlayers 2 --hidden 256 --epoch 1  --model CachedGCN  --nprocs 4 
# python main.py --dataset reddit  --nlayers 2 --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/cora/reddit_gcn.txt 


# arxiv
# python main.py --dataset ogbn-arxiv --nlayers 2 --hidden 64 --epoch 200  --model DecoupleGCN  --nprocs 4

# python main.py --dataset ogbn-arxiv --nlayers 2 --hidden 64 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/archive/archive.txt


#gat
# python main.py --dataset cora --nlayers 8 --hidden 128 --epoch 200  --model GAT --nprocs 4 

#20241120
# python main.py --dataset reddit  --nlayers 2 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk2_decouple.txt 
# python main.py --dataset reddit  --nlayers 4 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk4_decouple.txt 
# python main.py --dataset reddit  --nlayers 8 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk8_decouple.txt 
# python main.py --dataset reddit  --nlayers 4 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk4_decouple_test.txt 
# python main.py --dataset reddit  --nlayers 8 --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk8.txt 
# python main.py --dataset reddit  --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/reddit/reddit_gcnk4_origin.txt 

# python main.py --dataset ogbn-products  --nlayers 4 --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/products/products_gcnk4.txt 
# python main.py --dataset ogbn-products  --nlayers 8 --hidden 256 --epoch 200  --model CachedGCN  --nprocs 4 --savepath ./outputs/products/products_gcnk8.txt 
# python main.py --dataset ogbn-products  --nlayers 4 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/products/products_gcnk4_decouple.txt 
# python main.py --dataset ogbn-products  --nlayers 8 --hidden 256 --epoch 200  --model DecoupleGCN  --nprocs 4 --savepath ./outputs/products/products_gcnk8_decouple.txt 


# python main.py --dataset reddit  --hidden 256 --epoch 200  --model GAT  --nprocs 4 --savepath ./outputs/gat.txt 
# python main.py --dataset ogbn-products  --hidden 256 --epoch 200  --model GAT  --nprocs 4 --savepath ./outputs/gat.txt 

# python main.py --dataset ogbn-products  --hidden 256 --epoch 200 --nlayers 4 --model GCN  --nprocs 4 --savepath ./outputs/origin_gcn_products_k4.txt 
# python main.py --dataset reddit  --hidden 256 --epoch 200 --nlayers 2 --model GCN  --nprocs 4 --savepath ./outputs/TEST.txt 

# GAT可扩展性
# python main.py --dataset reddit  --hidden 128  --epoch 200  --model GAT  --nprocs 4 --savepath ./outputs/gat.txt
# python main.py --dataset reddit  --hidden 256 --epoch 200  --model GAT  --nprocs 3 --savepath ./outputs/gat.txt
# python main.py --dataset reddit  --hidden 256 --epoch 200  --model GAT  --nprocs 2 --savepath ./outputs/gat.txt
# python main.py --dataset reddit  --hidden 256 --epoch 200  --model GAT  --nprocs 1 --savepath ./outputs/gat.txt





# #超参数测试

# # 定义数组
# edges=("80M" "320M")
# features=(256 1024)
# labels=(16 64)
# trs=(0.1 0.5 0.8)

# # 遍历 edge 数组
# for e in "${edges[@]}"; do
#     dataset="e${e}_f512_l32_t0.5"
#     python main.py --dataset "${dataset}" --nlayers 2 --hidden 256 --epoch 200 --model CachedGCN --nprocs 4 >> ./log_random/e${e}_f512_l32_t0.5.log
#     echo "Finish ${dataset}"
# done

# # 遍历 feature 数组
# for f in "${features[@]}"; do
#     dataset="e160M_f${f}_l32_t0.5"
#     python main.py --dataset "${dataset}" --nlayers 2 --hidden 256 --epoch 200 --model CachedGCN --nprocs 4 >> ./log_random/e160M_f${f}_l32_t0.5.log
#     echo "Finish ${dataset}"
# done

# # 遍历 label 数组
# for l in "${labels[@]}"; do
#     dataset="e160M_f512_l${l}_t0.5"
#     python main.py --dataset "${dataset}" --nlayers 2  --hidden 256 --epoch 200 --model CachedGCN --nprocs 4 >> ./log_random/e160M_f512_l${l}_t0.5.log
#     echo "Finish ${dataset}"
# done

# # 遍历 tr 数组
# for t in "${trs[@]}"; do
#     dataset="e160M_f512_l32_t${t}"
#     python main.py --dataset "${dataset}" --hidden 256 --epoch 200 --model CachedGCN --nprocs 4 >> ./log_random/e160M_f512_l32_t${t}.log
#     echo "Finish ${dataset}"
# done




#20241219
# python main.py --dataset reddit  --nlayers 2 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt 
# python main.py --dataset reddit  --nlayers 4 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt 
# python main.py --dataset reddit  --nlayers 8 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt

# python main.py --dataset ogbn-products  --nlayers 2 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt  
# python main.py --dataset ogbn-products  --nlayers 4 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt 
# python main.py --dataset ogbn-products  --nlayers 8 --hidden 256 --epoch 200  --model GCN  --nprocs 4 --savepath ./outputs/test.txt 
