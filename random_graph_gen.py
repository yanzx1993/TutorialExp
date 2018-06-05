# 这里把全部Warning过滤掉了.
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# networkx是专门处理网络、图等数据结构的类.
import networkx as nx
import numpy as np
from numpy.random import random

# 颜色一般是每两位十六进制数代表一个0~255的整数, 表示一个颜色从最浅到最深。
# 次序一般是#RRGGBB，分别代表红色、绿色和蓝色。
COLORS = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
          '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']


def flip(p):
    return random() < p


def random_pairs(nodes, p):
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and flip(p):
                yield u, v


def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    edge_list = list(random_pairs(nodes, p))
    G.add_edges_from(edge_list)
    # print(nx.adjacency_matrix(G).todense())
    return G


def make_weighted_random_graph(n, p, weights):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    edge_list = list(random_pairs(nodes, p))
    if (weights == "normal"):
        weights = np.random.uniform(50, 100, len(edge_list))
    for i in range(len(edge_list)):
        G.add_edges_from([edge_list[i]], weight=weights[i])
    # print(nx.adjacency_matrix(G).todense())
    return G


def make_adj_matrix(G):
    return nx.to_numpy_matrix(G)


def make_laplacian_matrix(G):
    a = nx.normalized_laplacian_matrix(G)
    # print(a.todense())
    return a.todense()


def make_laplacian_list(graph_topology, node_size, orders):
    if graph_topology is None:
        print("Network topology is not initialized yet")
        return 0
    graph_laplacian_list = list()
    graph_laplacian_list.append(np.identity(node_size))
    lap = make_laplacian_matrix(graph_topology)
    base = make_laplacian_matrix(graph_topology)
    graph_laplacian_list.append(np.asarray(make_laplacian_matrix(graph_topology)))
    if orders > 2:
        for i in range(2, orders):
            lap = np.matmul(lap, base)
            graph_laplacian_list.append(np.asarray(lap))
    return graph_laplacian_list


def all_shortest_paths(G):
    return list(nx.all_pairs_shortest_path(G))


def simple_paths(G, source, target, cutoff):
    return list(nx.all_simple_paths(G, source, target, cutoff))


random_graph = make_random_graph(100, 0.5)
# print(all_shortest_paths(random_graph)[3][1])
# print(simple_paths(random_graph,0,2,3))
nx.draw_circular(random_graph,
                 node_color=COLORS[3],
                 node_size=1000,
                 with_labels=True)

# plt.show()
matrix = nx.to_numpy_matrix(random_graph)
