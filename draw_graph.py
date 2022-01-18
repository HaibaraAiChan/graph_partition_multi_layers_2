import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy
import torch
from pyvis.network import Network

matplotlib.use('Agg')


def merge_(list1, list2):
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
	return merged_list


def draw_graph(G, epoch):
    
	fig = plt.figure()
	black_edges = G.edges()
	black_edges = list(black_edges)
	print(black_edges[0])
	print(black_edges[1])
	# print('total eid number   '	)
	# print(len(black_edges[0]))
	# dd = int(len(black_edges[0]) / 2)
	dd = int(len(black_edges[0]))
	black_edges[0] = black_edges[0].tolist()
	black_edges[1] = black_edges[1].tolist()
	black_edges = merge_(black_edges[0][:dd], black_edges[1][:dd])
	# print('black_edges')
	# print(black_edges)
	nx_G = G.to_networkx()
	# nx_G = G.to_networkx().to_undirected()

	# pos = nx.kamada_kawai_layout(nx_G)
	pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	# nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	nx.draw_networkx_edges(nx_G, pos, edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	# plt.show()
	plt.savefig('TTTTTTTTT karate full batch sub-graph.eps',format='eps')
	return





def gen_pyvis_graph(G, epoch):
	print(G)
	sources_, targets_ = G.edges()
	sources=sources_.detach().numpy()
	targets=targets_.detach().numpy()
	edge_data = zip(sources, targets)
	pyvis_net = Network(notebook=True)
	for e in edge_data:
		
		src = int(e[0])
		dst = int(e[1])
		pyvis_net.add_node(1, label="Node 1")
		pyvis_net.add_node(src, label=str(src))
		pyvis_net.add_node(dst, label=str(dst))
		pyvis_net.add_edge(src, dst)
	file_name=str(epoch)+'_nx.html'
	pyvis_net.show(file_name)
	return 
    




def draw_nx(nx_G, epoch):
	fig = plt.figure()
	pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	# nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	# nx.draw_networkx_edges(nx_G,  edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	plt.show()
	plt.savefig('figures/'+str(epoch)+'_TTTTTTTTT karate full batch sub-graph_networkx.jpg',format='jpg')
	return
    
    
def dgl_to_networkx(g,epoch):
    if not g.is_homogeneous:
        raise DGLError('dgl.to_networkx only supports homogeneous graphs.')
    print(g)
    src, dst = g.edges()

    print()
    src = src.detach().numpy().astype(int)
    dst = dst.detach().numpy().astype(int)
    # xiangsx: Always treat graph as multigraph
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_nodes_from(range(g.number_of_nodes()))
    for eid, (u, v) in enumerate(zip(src, dst)):
        # print(eid)
        # nx_graph.add_node(u, label=str(u))
        # nx_graph.add_node(v, label=str(v))
        nx_graph.add_edge(int(u), int(v), id=int(eid))
    # nx.draw(nx_graph)
    draw_nx(nx_graph,epoch)
    
    return nx_graph


def generate_interactive_graph(G, epoch):
	
	
	nt = Network(notebook=True)
	nx_graph=dgl_to_networkx(G, epoch)
	# nx_graph = G.to_networkx()
	# nx_graph = nx.complete_graph(5)
	nt.barnes_hut()
	nt.show_buttons(filter_=['physics'])

	nt.from_nx(nx_graph)
	file_name='figures/'+str(epoch)+'_nx.html'
	nt.show(file_name)
	
