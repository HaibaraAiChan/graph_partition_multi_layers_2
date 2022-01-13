import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean

block_to_graph=None
args=None
ideal_partition_size=0

def InitializeBitList(p_list):
    
    nums_p=len(p_list); 
    bit_dict={}
    for i in range (nums_p-1):
        A = p_list[i]
        B = p_list[i + 1]; 
        for k in A:
            bit_dict[k] = 0
        for m in B:
            bit_dict[m] = 1
        
    return bit_dict
    
def redundancy_check(cur_part_in_size,ideal_part_in_size,target_redun):
    redundancy_flag=True
    r = cur_part_in_size/ideal_part_in_size
    if r > target_redun:
        redundancy_flag=False
    return redundancy_flag
    
def balance_check_2(A_in,B_in,alpha):
    balance_flag=True
    avg = (len(A_in)+len(B_in))/2
    if abs(len(B_in)-len(A_in)) >alpha*avg:
        balance_flag=False
    return balance_flag
    
def get_Red_Rate(bit_dict): 
    global ideal_partition_size

    A_o,B_o=get_two_partition_seeds(bit_dict)
    len_A,len_B=get_src_nodes_len(block_to_graph,A_o,B_o)
    rate =0
    ratio_A = len_A/ideal_partition_size
    ratio_B = len_B/ideal_partition_size
    # cost = max(ratio_A,ratio_B)
    rate = mean([ratio_A,ratio_B])
    return rate, ratio_A,ratio_B   #   minimize the max ratio of 
    
    
def getRedundancyRate(len_A, len_B, ideal_part_in_size):
    rate =0
    ratio_A = len_A/ideal_part_in_size
    ratio_B = len_B/ideal_part_in_size
    # cost = max(ratio_A,ratio_B)
    rate = mean([ratio_A,ratio_B])
    return rate, ratio_A,ratio_B   #   minimize the max ratio of 
    
def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	# print('current mini batch size of output nodes ', mini_batch)
	return mini_batch
	
def gen_batch_output_list(OUTPUT_NID,indices,mini_batch):
	
	map_output_list = list(numpy.array(OUTPUT_NID)[indices])
		
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
			
	output_num = len(OUTPUT_NID)
	
	# print(batches_nid_list)
	weights_list = []
	for i in batches_nid_list:
		# temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list
	
	
def random_shuffle(full_len):
	indices = numpy.arange(full_len)
	numpy.random.shuffle(indices)
	return indices

def gen_batched_seeds_list(OUTPUT_NID, args):
	'''

	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------

	'''
	selection_method = args.selection_method
	num_batch = args.num_batch
	# mini_batch = 0
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	mini_batch_size=get_mini_batch_size(full_len,num_batch)
	args.batch_size=mini_batch_size
	if selection_method == 'range_init_graph_partition' :
		indices = [i for i in range(full_len)]
	
	if selection_method == 'random_init_graph_partition' :
		indices = random_shuffle(full_len)
		
	elif selection_method == 'similarity_init_graph_partition':
		indices = torch.tensor(range(full_len)) #----------------------------TO DO
	
	batches_nid_list, weights_list=gen_batch_output_list(OUTPUT_NID,indices,mini_batch_size)
	
	return batches_nid_list, weights_list


    



def get_weight_list(batched_seeds_list):
    
    output_num = len(sum(batched_seeds_list,[]))
    # print(output_num)
    weights_list = []
    for seeds in batched_seeds_list:
		# temp = len(i)/output_num
        weights_list.append(len(seeds)/output_num)
    return weights_list



def exists(it):
    return (it is not None)

def get_two_partition_seeds(bit_dict):
    
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    return  A_o, B_o
    


    
    
def counting(bit_dict, side):
    seeds=[k for k in bit_dict if bit_dict[k] == int(side)]
    return len(seeds)



def balance_checking(block_to_graph,bit_dict,alpha): 
    
    flag=False
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    print('\t\t\t length of A_o,B_o  '+str(len(A_o))+' '+str(len(B_o)))
          
    len_A_part,len_B_part = get_src_nodes_len(block_to_graph,A_o,B_o)
    len_=len_A_part+len_B_part
    avg = len_/2
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*alpha:
        flag=True
    return flag, len_A_part,len_B_part  

def move_group_nids_balance_redundancy_check(block_to_graph,bit_dict, nids,alpha, red_rate,ideal_part_in_size):
    for nid in nids:
        bit_dict[nid]=1
        
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    for nid in nids:
        bit_dict[nid]=1-bit_dict[nid]
        
    balance_flag=False
    # t1=time.time()
    len_A_part,len_B_part = get_src_nodes_len(block_to_graph,A_o,B_o)
    # print('get_src_nodes_len ', time.time()-t1)

    len_=len_A_part+len_B_part
    avg = len_/2
    
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*alpha:
        balance_flag=True
    else:
        balance_flag=False

    # balance_flag=True    # now test 

    # red, ratio_A, ratio_B=getRedundancyRate(len_A_part, len_B_part, ideal_part_in_size)
    red, ratio_A, ratio_B=get_Red_Rate(bit_dict)
    red_flag=False
    if red< red_rate and ratio_A>=1 and ratio_B>=1:
        red_flag=True
        red_rate=red

    # red_flag=True if red<red_rate else False
    if red_flag:
        print(' redundancy '+str(red))

    return balance_flag and red_flag, red_rate
 
 

def calculate_redundancy(idx,i, A_o, B_o, side,locked_nodes):
    global block_to_graph
    gain=0
    in_nids=block_to_graph.predecessors(i).tolist()
    # print('in_nids')
    # print(in_nids)

    if side==0:
        gain_pos=len(list(set(in_nids).intersection(set(B_o))))
        gain_neg=len(list(set(in_nids).intersection(set(A_o)))) 
    else:
        gain_pos=len(list(set(in_nids).intersection(set(A_o))))
        gain_neg=len(list(set(in_nids).intersection(set(B_o)))) 

    gain=gain_pos-gain_neg 
    # print('gain ',gain)
    if gain>=0 and not locked_nodes[i] :
        return (idx,i)
    
    return (idx,None)
                

def updateRed_group(block_to_graph, bit_dict, side,locked_nodes, cur_rate, alpha,ideal_part_in_size):
    global args
    #------------------------
    # bit_dict=balance_check_and_exchange(bit_dict)
    #-------------------------
    ready_to_move=[]
    A_o,B_o=get_two_partition_seeds(bit_dict)
    if side==0:

        output=A_o
    else:
        output=B_o
        

    if len(output)>1: 
        if args.dataset=='karate':
            # ready_to_move=[]
            for idx, nid in enumerate(output):
                gain=0
                in_nids=block_to_graph.predecessors(nid).tolist()
                # print('nid \t', nid)
                # print('in_nids\t', in_nids)
                if side==0:
                    gain_pos=len(list(set(in_nids).intersection(set(B_o))))
                    gain_neg=len(list(set(in_nids).intersection(set(A_o)))) 
                else:
                    gain_pos=len(list(set(in_nids).intersection(set(A_o))))
                    gain_neg=len(list(set(in_nids).intersection(set(B_o)))) 


                gain=gain_pos-gain_neg 
                print('gain \t',gain)
                print()
                if gain>=0 and not locked_nodes[nid] :
                    ready_to_move.append(nid)
            print(ready_to_move)

            while len(ready_to_move)>1:
                flag,cur_rate=move_group_nids_balance_redundancy_check(block_to_graph,bit_dict, ready_to_move,alpha, cur_rate,ideal_part_in_size)
                if not flag :
                    # print('------------------------------- !!!!!!!!!!!!!! redundancy check failed, ')
                    gold_r=int(len(ready_to_move)*0.5)
                    ready_to_move=ready_to_move[:gold_r]
                else:
                    break

        else:
            pool = mp.Pool(mp.cpu_count())
            tmp_gains = pool.starmap_async(calculate_redundancy, [(idx, i, A_o, B_o,side, locked_nodes) for idx, i in enumerate(output)]).get()
            pool.close()
            ready_to_move = [list(r)[1] for r in tmp_gains]
            ready_to_move=list(filter(lambda v: v is not None, ready_to_move))

         

            while len(ready_to_move)>1:    
                flag,cur_rate = move_group_nids_balance_redundancy_check(block_to_graph,bit_dict, ready_to_move,alpha, cur_rate,ideal_part_in_size)
                if not flag :
                    print('------------------------------- !!!!!!!!!!!!!! redundancy check failed, ')
                    gold_r=int(len(ready_to_move)*0.2)
                    ready_to_move=ready_to_move[:gold_r]
                else:
                    
                    break

        if len(output)==len(ready_to_move):
            ready_to_move=ready_to_move[:len(output)-1]    

        print('the number of node ready to move is :', len(ready_to_move))

        for i in ready_to_move:
            locked_nodes[i]=True
        for nid in ready_to_move:
            bit_dict[nid]=1-side
    
    # A_o,B_o=get_two_partition_seeds(bit_dict)   
    # if len(A_o)==0 or len(B_o)==0:
    #     print()
    return    locked_nodes , bit_dict            
                 
                
    
    
def exchange_bit_dict_side(bit_dict):
    bit_dict={i: 1-bit_dict[i] for i in bit_dict}
    return bit_dict





def balance_check_and_exchange(bit_dict):
    global block_to_graph
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    len_A_part,len_B_part = get_src_nodes_len(block_to_graph,A_o,B_o)
    # if left partition size is less than right partition size, exchange side;
    # otherwise, do not exchange
    if len_B_part>0 and len_A_part>0 :
        if len_A_part-len_B_part < 0:
            bit_dict={i: 1-bit_dict[i] for i in bit_dict}
            
    return bit_dict
    

    
def get_partition_src_list_len(block_to_graph,batched_seeds_list,ideal_part_in_size):
    
    
    partition_src_list_len=[]
    redundancy_list=[]
    for seeds_nids in batched_seeds_list:
        in_nids = get_in_nodes(block_to_graph,seeds_nids)
        part=list(set(seeds_nids+in_nids))
        partition_src_list_len.append(len(part))
        redundancy_list.append(len(part)/ideal_part_in_size)
    return partition_src_list_len,redundancy_list



    
def get_src_nodes(block_to_graph,seeds_1,seeds_2):
    in_ids_1=list(block_to_graph.in_edges(seeds_1))[0].tolist()
    src_1= list(set(in_ids_1+seeds_1))
    in_ids_2=list(block_to_graph.in_edges(seeds_2))[0].tolist()
    src_2= list(set(in_ids_2+seeds_2))
    return src_1, src_2


def get_src_nodes_len(block_to_graph,seeds_1,seeds_2):

    in_ids_1=list(block_to_graph.in_edges(seeds_1))[0].tolist()
    src_len_1= len(list(set(in_ids_1+seeds_1)))
    in_ids_2=list(block_to_graph.in_edges(seeds_2))[0].tolist()
    src_len_2= len(list(set(in_ids_2+seeds_2)))
    
    return src_len_1, src_len_2


def get_in_nodes(block_to_graph,seeds):
    
    in_ids=list(block_to_graph.in_edges(seeds))[0].tolist()
    in_ids= list(set(in_ids))
    return in_ids
    
def update_Batched_Seeds_list(batched_seeds_list, bit_dict,i, j):

    batch_i=[k for k in bit_dict if bit_dict[k]==0]
    batch_j=[k for k in bit_dict if bit_dict[k]==1]
    
    batched_seeds_list.remove(batched_seeds_list[i])
    batched_seeds_list.insert(i,batch_i)
    
    batched_seeds_list.remove(batched_seeds_list[j])
    batched_seeds_list.insert(j,batch_j)

    return batched_seeds_list
    

def print_len_of_batched_seeds_list(batched_seeds_list):
    batch_list_str=''
    len_=''
    for i in batched_seeds_list:
        batch_list_str=batch_list_str+str(i)+', '
        len_=len_+str(len(i))+', '
    # batch_list_str=' '.join(batched_seeds_list)
    print('batch_seeds_list len ')
    # print(batch_list_str) 
    print(len_)
    
def print_len_of_partition_list(partition_src_list_len):
    # batch_list_str=''
    len_=''
    for i in partition_src_list_len:
        # batch_list_str=batch_list_str+str(i)+', '
        
        len_=len_+str(i)+', '
    # batch_list_str=' '.join(batched_seeds_list)
    print('partition_src_list len ')
    # print(batch_list_str) 
    print(len_)
    

def walk_terminate_1(g,bit_dict,red_rate, args,ideal_part_in_size):
    global block_to_graph

    redundancy_tolarent_steps=args.redundancy_tolarent_steps
    
    bestRate =red_rate
    best_bit_dict=bit_dict
    print('\twalk terminate 1')
    
    bit_dict=balance_check_and_exchange(bit_dict) 
    side=0
    # if left partition size is smaller than right partition, exchange them
    # make sure the left partition size is larger or equal with right partition size.
    # then, we can only focus the first step move nodes from the larger partition to the smaller one.
    A_o,B_o= get_two_partition_seeds(bit_dict) # get A_o, B_o based on global variable bit_dict
    subgraph_o = A_o+B_o
    locked_nodes_org={id:False for id in subgraph_o}
    # steps_=redundancy_tolarent_steps
    steps_=4
    locked_nodes=locked_nodes_org
    # t_b=time.time()
    for i in range(steps_):
        
        # tt=time.time()
        locked_nodes,bit_dict=updateRed_group(block_to_graph,bit_dict,side,locked_nodes,red_rate,args.alpha,ideal_part_in_size)
        # tt_e=time.time()
        if i % 1==0:
            # print('\t\t\tone update redundancy spend time _*_\t', tt_e-tt)	
            print('\t\t  --------------------------------------------group redundancy rate update  step :'+ str(i)+'  side '+str(side))
        tmpRate,ratio_A,ratio_B=get_Red_Rate(bit_dict)
        # tmpRate,ratio_A,ratio_B =getRedundancyRate(len_A_part,len_B_part,ideal_part_in_size)
        print('\t\t\t redundancy rate (ration_mean, ratio_A, ratio_B): '+str(tmpRate)+',  '+str(ratio_A)+',  '+ str(ratio_B))
        if tmpRate < bestRate: 
            bestRate = tmpRate
            best_bit_dict = bit_dict

        side=1-side
    # t_e=time.time()
    # print('\t\tupdate redundancy of segment  ', t_e-t_b)
    if (bestRate < red_rate) : #is there improvement? Yes
        tmpRate,ratio_A,ratio_B=get_Red_Rate(bit_dict)
         
        bit_dict = best_bit_dict
        rate = bestRate
        return True, bestRate, bit_dict

    if ratio_A > ratio_B:
        best_bit_dict=balance_check_and_exchange(best_bit_dict)
    return False, red_rate, bit_dict
    




def graph_partition_variant(batched_seeds_list, block_to_graph_global, args_o):
    global block_to_graph
    global args
    global ideal_partition_size
    args = args_o

    bit_dict={}
    print('----------------------------graph partition start---------------------')
    
    full_batch_graph_nids_size=len(block_to_graph_global.srcdata['_ID'])
    ideal_partition_size=(full_batch_graph_nids_size/args.num_batch)
   
    # full_batch_seeds = block_2_graph.dstdata['_ID'].tolist()
    num_batch=args.num_batch
    # balance_flag = False
    
    # print(list(block_2_graph.edges()))
    src_ids=list(block_to_graph_global.edges())[0]
    dst_ids=list(block_to_graph_global.edges())[1]
    local_g = dgl.graph((src_ids, dst_ids))
    local_g = dgl.remove_self_loop(local_g)
    # from draw_graph import draw_graph
    # draw_graph(local_g)
    block_to_graph=local_g

    print('before graph partition ')
    src_len_list,_ = get_partition_src_list_len(local_g, batched_seeds_list,ideal_partition_size)
    print_len_of_partition_list(src_len_list) 

    print('{}-'*40)
    print()

    i=0
    for i in range(num_batch-1):# no (end, head) pair
        print('-------------------------------------------------------------  compare batch pair  (' +str(i)+','+str(i+1)+')')
        tii=time.time()

        print_len_of_batched_seeds_list(batched_seeds_list)    
        A_o=batched_seeds_list[i]
        B_o=batched_seeds_list[i+1]
        len_A_part,len_B_part = get_src_nodes_len(local_g,A_o,B_o)
        
        tij=time.time()
        print('\n\tpreparing two sides time : ' , time.time()-tii)
        bit_dict=InitializeBitList([A_o,B_o])
        print('\tInitialize BitList time : ' , time.time()-tij)
        tik=time.time()
        red_rate,ratio_A,ratio_B =getRedundancyRate(len_A_part,len_B_part,ideal_partition_size) #r_cost=max(r_A, r_B)
        print('\tgetRedundancyCost: time  ' , time.time()-tik)
        print()
        print('\t\t\t\t\tlength of partitions '+ str(len_A_part)+', '+str(len_B_part))
        print()
        if red_rate < 1.0:
            continue
        
        # tih=time.time()
        # print('\t'+'-'*80)
        # max_neighbors=int(args.fan_out) # only for 1-layer model
        # local_nids=list(bit_dict.keys())
        # local_in_degrees = block_to_graph.in_degrees(local_nids).tolist()
        # max_in_degree=max(local_in_degrees) # only consider output nodes in degrees
        # maxDegree= max(max_neighbors, max_in_degree)
        # print('\t\t\t\tmaxDegree ',maxDegree)
        
        print('\tbefore terminate 1 the redundancy rate: ', red_rate)
        print('\t'+'-'*80)
        pass_=2
        for pass_step in range(pass_):
            if args.walkterm==1:
                ti=time.time()
                improvement,red_rate,bit_dict=walk_terminate_1(local_g,bit_dict,red_rate, args,ideal_partition_size)
                print('\twalk terminate 1 spend time', time.time()-ti)
                print('\tafter terminate 1, whether it has improvement: ',improvement)
                print('\tafter terminate 1 the  redundancy rate ',red_rate)
                if not improvement: # go several passes, until there is no improvement, 
                    batched_seeds_list=update_Batched_Seeds_list(batched_seeds_list, bit_dict,i, i+1)
                    # update batched_seeds_list based on bit_dict
                    print('\tpass '+str(pass_step)+'  ')
                    src_len_list,_ = get_partition_src_list_len(local_g, batched_seeds_list,ideal_partition_size)
                    print_len_of_partition_list(src_len_list) 
                    # print out the result after exchange batched_seeds_list
                    
                    print()
                    break
            
        #--------------------- initialization checking done   ----------------   
        print('\t'+'-'*50 +'end of batch '+ str(i))
    
   
    weight_list=get_weight_list(batched_seeds_list)
    len_list,redundancy_list=get_partition_src_list_len(local_g, batched_seeds_list,ideal_partition_size)
    print('after graph partition')
    return batched_seeds_list, weight_list, len_list
 
def global_2_local(block_to_graph,batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    # local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    
    t1=time.time()
    local_batched_seeds_list=[]
    for global_in_nids in batched_seeds_list:
        tt=time.time()
        local_in_nids = list(map(global_nid_2_local.get, global_in_nids))
        local_batched_seeds_list.append(local_in_nids)
    return local_batched_seeds_list

def local_2_global(block_to_graph,local_batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    # global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    
    t1=time.time()
    global_batched_seeds_list=[]
    for local_in_nids in local_batched_seeds_list:
        tt=time.time()
        global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
        global_batched_seeds_list.append(global_in_nids)
    return global_batched_seeds_list	
	

def  random_init_graph_partition(block_to_graph, args):
    tt = time.time()
    OUTPUT_NID, _ = torch.sort(block_to_graph.ndata[dgl.NID]['_N_dst'])
    batched_seeds_list,_ = gen_batched_seeds_list(OUTPUT_NID, args)
    
    t1 = time.time()
    batched_seeds_list=global_2_local(block_to_graph, batched_seeds_list) # global to local
    print('transfer time: ', time.time()-t1)

    #The graph_parition is run in block to graph local nids,it has no relationship with raw graph
    batched_seeds_list,weights_list, p_len_list=graph_partition_variant(batched_seeds_list, block_to_graph, args)
    batched_seeds_list=local_2_global(block_to_graph, batched_seeds_list) # local to global
    print('graph partition total spend ', time.time()-t1)
    t2=time.time()-tt
    return batched_seeds_list, weights_list,t2, p_len_list