import torch
import dgl
import numpy
import time
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array
from graph_partitioner import Graph_Partitioner
from my_utils import gen_batch_output_list
from draw_graph import draw_graph, generate_interactive_graph

def unique_tensor_item(combined):
	uniques, counts = combined.unique(return_counts=True)
	return uniques.type(torch.long)


def unique_edges(edges_list):
	temp = []
	for i in range(len(edges_list)):
		tt = edges_list[i]  # tt : [[],[]]
		for j in range(len(tt[0])):
			cur = (tt[0][j], tt[1][j])
			if cur not in temp:
				temp.append(cur)
	# print(temp)   # [(),(),()...]
	res_ = list(map(list, zip(*temp)))  # [],[]
	res = tuple(sub for sub in res_)
	return res


def generate_random_mini_batch_seeds_list(OUTPUT_NID, args):
	'''
	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------
	'''
	selection_method = args.selection_method
	mini_batch = args.batch_size
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	if selection_method == 'random':
		indices = torch.randperm(full_len)  # get a permutation of the index of output nid tensor (permutation of 0~n-1)
	else: #selection_method == 'range'
		indices = torch.tensor(range(full_len))

	output_num = len(OUTPUT_NID.tolist())
	map_output_list = list(numpy.array(OUTPUT_NID)[indices.tolist()])
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
	weights_list = []
	for i in batches_nid_list:
		temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list


def get_global_graph_edges_ids_2(raw_graph, cur_subgraph):
	src = cur_subgraph.edges()[0]
	dst = cur_subgraph.edges()[1]
	
	src = src.long()
	dst = dst.long()
	
	raw_src, raw_dst = cur_subgraph.ndata[dgl.NID]['_N_src'][src], cur_subgraph.ndata[dgl.NID]['_N_dst'][dst]

	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


def get_global_graph_edges_ids(raw_graph, cur_block):
	'''
		Parameters
		----------
		raw_graph : graph
		cur_block: (local nids, local nids): (tensor,tensor)

		Returns
		-------
		global_graph_edges_ids: []                    current block edges global id list
	'''

	src, dst = cur_block.all_edges(order='eid')
	src = src.long()
	dst = dst.long()
	# print(src.tolist())
	# print(dst.tolist())
	raw_src, raw_dst = cur_block.srcdata[dgl.NID][src], cur_block.dstdata[dgl.NID][dst]
	# print(raw_src.tolist())
	# print(raw_dst.tolist())
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)


def generate_one_block(raw_graph, global_eids, global_srcnid):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	_graph = dgl.edge_subgraph(raw_graph, global_eids)
	edge_dst_list = _graph.edges()[1].tolist()
	dst_local_nid_list = list(set(edge_dst_list))
	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))

	global_nid_list = _graph.ndata[dgl.NID].tolist()
	block_nid_list = new_block.ndata[dgl.NID]['_N'].tolist()
	block_dst_nid_list = new_block.dstdata[dgl.NID].tolist()

	final_nid_list = [global_nid_list[i] for i in block_nid_list]  # mapping global graph nid <--- block local nid
	final_dst_nid_list = [global_nid_list[i] for i in block_dst_nid_list]

	new_block.ndata[dgl.NID] = {'_N': torch.tensor(final_nid_list, dtype=torch.long)}
	new_block.dstdata[dgl.NID] = torch.tensor(final_dst_nid_list, dtype=torch.long)
	new_block.srcdata[dgl.NID] = torch.tensor(final_nid_list, dtype=torch.long)

	return new_block

def check_connections_0(batched_nodes_list, current_layer_subgraph):
	res=[]
	
	# multi-layers model: current_layer_subgraph, here
	src_nid_list = current_layer_subgraph.srcdata['_ID'].tolist()
	# print('\n *************************************   src nid of current layer subgraph')
	# print(src_nid_list)
	dict_nid_2_local = {src_nid_list[i]: i for i in range(0, len(src_nid_list))}
	# print('\n *************************************   dict_nid_2_local')
	eids_global_list = current_layer_subgraph.edata['_ID'].tolist()
	

	for step, output_nid in enumerate(batched_nodes_list):
		# print(step, ' -----------------------------------------------step ')
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		given_nid_list_ = output_nid
		# given_nid_list_ = output_nid.tolist()
		local_given_output_nids = list(map(dict_nid_2_local.get, given_nid_list_))
		local_in_edges_tensor = current_layer_subgraph.in_edges(local_given_output_nids, form='all')

		# get local srcnid and dstnid from subgraph
		mini_batch_srcid_local_list = list(local_in_edges_tensor)[0].tolist()
		srcid_list = list(numpy.array(src_nid_list)[mini_batch_srcid_local_list])
		# map local srcnid , dstnid,  eid to global
		eid_local_list = list(local_in_edges_tensor)[2]
		eid_list = list(numpy.array(eids_global_list)[eid_local_list.tolist()])
		global_eid_tensor = torch.tensor(eid_list, dtype=torch.long)
		srcid = torch.tensor(list(set(given_nid_list_+ srcid_list)), dtype=torch.long)
		

		res.append((srcid, output_nid, global_eid_tensor))
	# print('res----------------------------------------')
	# print(len(shared_list))
	return res


def generate_blocks_for_one_layer(raw_graph, block_2_graph, batches_nid_list):

	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_0(batches_nid_list, block_2_graph)
	t2 = time.time()
	check_connection_time.append(t2-t1) #------------------------------------------
	src_list=[]
	dst_list=[]

	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
		# print('batch ' + str(step) + '-' * 30)
		t_ = time.time()
		cur_block = generate_one_block(raw_graph, current_block_global_eid, srcnid)
		t__=time.time()
		block_generation_time.append(t__-t_)  #------------------------------------------
		
		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

		# data_loader.append((srcnid, dstnid, [cur_block]))
		
	# print("\nconnection checking time " + str(sum(check_connection_time)))
	# print("total of block generation time " + str(sum(block_generation_time)))
	# print("average of block generation time " + str(mean(block_generation_time)))
	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	mean_block_gen_time = mean(block_generation_time)


	return blocks, src_list,dst_list,(connection_time, block_gen_time, mean_block_gen_time)



def generate_dataloader_w_partition(raw_graph, block_to_graph_list, args):
	for layer, block_to_graph in enumerate(block_to_graph_list):
		
		current_block_eidx, current_block_edges = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
		block_to_graph.edata['_ID'] = current_block_eidx
		if layer == 0:
			my_graph_partitioner=Graph_Partitioner(block_to_graph, args) #init a graph partitioner object
			batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list=my_graph_partitioner.init_graph_partition()

			print('partition_len_list')
			print(p_len_list)
			args.batch_size=my_graph_partitioner.batch_size
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, batched_output_nid_list)
			connection_time, block_gen_time, mean_block_gen_time = time_1
			# batch_list_generation_time = t1 - tt
			time_2 = (connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time)
		else:
			return
	data_loader=[]
	# TODO
	return data_loader, weights_list, time_2

def gen_grouped_dst_list(prev_layer_blocks):
	post_dst=[]
	for block in prev_layer_blocks:
		src_nids = block.srcdata['_ID'].tolist()
		post_dst.append(src_nids)
	return post_dst # return next layer's dst nids(equals prev layer src nids)

def generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args):
	data_loader=[]
	weights_list=[]
	num_batch=0
	blocks_list=[]
	final_dst_list =[]
	final_src_list=[]
	prev_layer_blocks=[]
	t_2_list=[]
	# prev_layer_src_list=[]
	# prev_layer_dst_list=[]
	for layer, block_to_graph in enumerate(block_to_graph_list):
		if layer ==0:
			current_block_eidx, current_block_edges = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
			block_to_graph.edata['_ID'] = current_block_eidx
			dst_nids=block_to_graph.dstdata['_ID'].tolist()
			# src_nids=block_to_graph.srcdata['_ID'].tolist()
			# print('time of batches_nid_list generation : ' + str(t1 - tt) + ' sec')
			t1=time.time()
			indices = [i for i in range(len(dst_nids))]
			batched_output_nid_list, w_list=gen_batch_output_list(dst_nids,indices,args.batch_size)
			tt=time.time()
			weights_list=w_list
			num_batch=len(batched_output_nid_list)
			print('layer ', layer)
			print('\tselection method range initialization spend ', time.time()-t1)
			# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
			blocks, src_list, dst_list,time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, batched_output_nid_list)
			connection_time, block_gen_time, mean_block_gen_time = time_1
			batch_list_generation_time = tt - t1
			time_2 = [connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time]
			t_2_list.append(time_2)
			prev_layer_blocks=blocks
			# prev_layer_dst_list=dst_list
			# prev_layer_src_list=src_list

			blocks_list.append(blocks)
			final_dst_list=dst_list

		else:
			current_block_eidx, current_block_edges = get_global_graph_edges_ids_2(raw_graph, block_to_graph)
			block_to_graph.edata['_ID'] = current_block_eidx
			output_nids=block_to_graph.dstdata['_ID']
			# print('time of batches_nid_list generation : ' + str(t1 - tt) + ' sec')
			t1=time.time()
			
			grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
			tt=time.time()
			print('layer ',layer)
			print('\tselection method range initialization spend ', time.time()-t1)
			
			blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer(raw_graph, block_to_graph, grouped_output_nid_list)
			connection_time, block_gen_time, mean_block_gen_time = time_1
			batch_list_generation_time = tt-t1
			time_2 = [connection_time, block_gen_time, mean_block_gen_time, batch_list_generation_time]
			t_2_list.append(time_2)

			if layer<args.num_layers-1:
				prev_layer_blocks=blocks
				# prev_layer_dst_list=dst_list
				# prev_layer_src_list=src_list
			else:
				final_src_list=src_list
			blocks_list.append(blocks)

	for bid in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][bid])
		dst = final_dst_list[bid]
		src = final_src_list[bid]
		data_loader.append((src, dst, cur_blocks))

	sum_list=[]
	for i in range(0,len(t_2_list),2):
		list1=t_2_list[i]
		list2=t_2_list[i+1]
		for (item1, item2) in zip(list1, list2):
			sum_list.append(item1+item2)

	return data_loader, weights_list, sum_list
		

def generate_dataloader(raw_graph, block_to_graph_list, args):
    
    
    if 'partition' in args.selection_method:
        return generate_dataloader_w_partition(raw_graph, block_to_graph_list, args)
    else:
        return generate_dataloader_wo_gp_Pure_range(raw_graph, block_to_graph_list, args)
		# return generate_dataloader_0(raw_graph, block_to_graph, args)
