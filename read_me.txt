pseudo_gen_same_full_batch_subgraph.py
    generate full batch subgraph from raw graph based on sampling fan out

pseudo_reddit_same_subgraph_gp_wo.py
pseudo_karate_same_subgraph.py
pseudo_cora_same_subgraph.py
    pseudo mini batch train of dataset without graph partition

pseudo_reddit_gp_w.py
pseudo_cora_gp_w.py
pseudo_karate.py
    pseudo mini batch train of dataset with graph partition (redundancy reduction)


draw_graph.py


block_dataloader_graph.py
    block dataloader transfer as DGLGraph instead of DGLBlock, 
    This block dataloader will be called by pseudo mini batch train. 

graph_partitioner.py
    if we choose to do redundancy reduction, 
    the block data loader will call graph partitioner to reduce redundancy based on
    the balance alpha.


