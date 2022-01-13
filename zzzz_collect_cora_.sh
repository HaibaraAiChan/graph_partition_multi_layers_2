#!/bin/bash

File=pseudo_reddit_gp_w.py
# Data=(cora reddit ogbn-products ogbn-mag)
Data1=ogbn-products
Data2=ogbn-mag
Aggre=mean

ratio=(0.95 0.9 0.5 0.2)

# mkdir logs

for rat in ${ratio[@]}
do
    # python $File \
    #     --dataset $Data1 \
    #     --aggre $Aggre \
    #     --selection-method balanced_init_graph_partition \
    #     --balanced_init_ratio $rat \
    #     --num-epochs 6 \
    #     --eval-every 5 &> logs/{$Data1}_{$Aggre}_pseudo_balance_init_ratio_${rat}.log

    python $File \
        --dataset $Data2 \
        --aggre $Aggre \
        --selection-method balanced_init_graph_partition \
        --balanced_init_ratio $rat \
        --num-epochs 6 \
        --eval-every 5 &> logs/${Data2}_${Aggre}_pseudo_balance_init_ratio_${rat}.log

done

# python $File \
#     --dataset $Data1 \
#     --aggre $Aggre \
#     --selection-method random_init_graph_partition \
#     --alpha 0.5 \
#     --num-epochs 6 \
#     --eval-every 5 &> logs/${Data1}_${Aggre}_alpha_1.0_pseudo_random_init.log

python $File \
    --dataset $Data2 \
    --aggre $Aggre \
    --selection-method random_init_graph_partition \
    --alpha 0.5 \
    --num-epochs 6 \
    --eval-every 5 &> logs/${Data2}_${Aggre}_alpha_1.0_pseudo_random_init.log








#python pseudo_gen_same_full_batch_subgraphs.py
# File=pseudo_ogbn_mag_red.py
# Data=ogbn-mag
# Aggre=lstm
