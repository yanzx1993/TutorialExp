# 这里把全部Warning过滤掉了.
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# networkx是专门处理网络、图等数据结构的类.
import networkx as nx
import numpy as np
from numpy.random import random
import random_graph_gen as rd


# 颜色一般是每两位十六进制数代表一个0~255的整数, 表示一个颜色从最浅到最深。
# 次序一般是#RRGGBB，分别代表红色、绿色和蓝色。


class NetworkTopology(object):
    def __init__(self, name, node_size=100, link_probability=0.5, weights=None, graph_type=0,
                 imported_graph=None):
        self.name = name
        self.node_rank_list = list()
        self.node_size = node_size
        if imported_graph is not None:
            self.graph_topology = imported_graph
        else:
            self.graph_topology = self.generate_network_topology(node_size, link_probability, weights, graph_type)
        self.attribute_list = self.initialize_node_attributes()
        self.max_attribute_list = self.get_max_attribute_list()

    def generate_network_topology(self, nodes, link_probability, weights=None, types=0):
        self.node_size = nodes
        if types == 0:
            if weights is None:
                self.graph_topology = rd.make_random_graph(nodes, link_probability)
            else:
                self.graph_topology = rd.make_weighted_random_graph(nodes, link_probability, weights)
            return self.graph_topology
        elif types == 1:
            return 0

    def initialize_node_attributes(self, type="normal", customize=None):
        attribute_list = list()
        if (type == "normal"):
            attribute_list.append({"name": "cpu_in_use", "attributes": np.zeros([self.node_size])})
            attribute_list.append({"name": "bandwidth_in_use", "attributes": np.zeros([self.node_size])})
            attribute_list.append({"name": "cpu_max", "attributes": np.random.uniform(50, 200, self.node_size)})
            bandwidth_max = np.zeros([self.node_size])
            # for i in range (self.node_size):
            #     for j in range (self.node_size):
            #         #bandwidth_max[i]=bandwidth_max[i]+self.graph_topology.get_edge_data(i,j).values()[0]
            #         print(self.graph_topology.get_edge_data(i,j))
            #         print(self.graph_topology.get_edge_data(j, i))
            for (u, v) in self.graph_topology.edges():
                bandwidth_max[u - 1] += self.graph_topology.get_edge_data(u, v)["weight"]
                bandwidth_max[v - 1] += self.graph_topology.get_edge_data(u, v)["weight"]
            attribute_list.append({"name": "bandwidth_max", "attributes": bandwidth_max})
            # attribute_list.append({"name": "cpu_remaining", "attributes":attribute_list[2]["attributes"]-attribute_list[0]["attributes"]})
            attribute_list.append({"name": "current_embedding", "attributes": np.zeros([self.node_size])})
        return attribute_list

    def set_node_attributes(self, attributes, name):
        if self.graph_topology is None:
            print("Network topology is not initialized yet")
            return 0
        nx.set_node_attributes(self.graph_topology, attributes, name)
        for i in range(len(self.attribute_list)):
            if self.attribute_list[i]["name" == name]:
                self.attribute_list[i]["attributes"] = attributes
                return self.attribute_list
        self.attribute_list.append({"name": name, "attributes": attributes})
        return self.attribute_list

    def get_node_attributes(self, name=None):
        if name is None:
            return self.attribute_list
        else:
            for i in range(len(self.attribute_list)):
                if self.attribute_list[i]['name'] == name:
                    return self.attribute_list[i]
                else:
                    print("node attributes with name %s do not exist" % name)
                    return 0

    def get_max_attribute_list(self):
        self.max_attribute_list = list()
        for i in range(len(self.attribute_list)):
            self.max_attribute_list.append(
                {"name": self.attribute_list[i]["name"], "max": max(self.attribute_list[i]["attributes"])})
        return self.max_attribute_list

    def node_rank(self, attribute_name=None):
        attribute = None
        for i in range(len(self.attribute_list)):
            if (self.attribute_list[i]["name"] == attribute_name):
                attribute = self.attribute_list[i]["attributes"]
        if attribute is not None:
            idx = np.arange(0, len(attribute))
            a = dict(zip(idx, attribute))
        rank_list = nx.pagerank(self.graph_topology, personalization=a, weight="weight")
        self.node_rank_list = sorted(rank_list.items(), key=lambda items: items[1], reverse=True)
        return self.node_rank_list


class SubstrateNetwork(NetworkTopology):
    def __init__(self, name, node_size=100, link_probability=0.5, weights=None, graph_type=0, imported_graph=None):
        super(SubstrateNetwork, self).__init__(name, node_size, link_probability, weights, graph_type, imported_graph)
        self.name = name
        self.node_size = node_size
        self.resource_in_use = np.zeros([node_size])
        self.shortest_paths = self.all_shortest_paths()
        self.temporary_substrate_network = None
        self.temporary_embedding_result = np.zeros([node_size])
        self.graph_laplacian_list = self.generate_graph_laplacian_list(node_size, orders=3)

    def generate_graph_laplacian_list(self, node_size, orders):
        if self.graph_topology is None:
            print("Network topology is not initialized yet")
            return 0
        self.graph_laplacian_list = list()
        self.graph_laplacian_list.append(np.identity(node_size))
        lap = rd.make_laplacian_matrix(self.graph_topology)
        base = rd.make_laplacian_matrix(self.graph_topology)
        self.graph_laplacian_list.append(rd.make_laplacian_matrix(self.graph_topology))
        if orders > 2:
            for i in range(2, orders):
                lap = np.matmul(lap, base)
                self.graph_laplacian_list.append(lap)
        return self.graph_laplacian_list

    def all_shortest_paths(self):
        self.shortest_paths = list(nx.all_pairs_shortest_path(self.graph_topology))
        return self.shortest_paths

    def simple_paths(self, source, target, cutoff=5):
        return list(nx.edge_disjoint_paths(self.graph_topology, source, target, cutoff=cutoff))

    def copy_substrate_network(self):
        self.temporary_substrate_network = self.graph_topology
        return self.temporary_substrate_network

    def update_substrate_network(self, succeed=True):
        if (succeed == True):
            # self.attribute_list = self.temporary_substrate_network["attribute_list"]
            self.graph_topology = self.temporary_substrate_network
        self.temporary_embedding_result = np.zeros([self.node_size])


class VirtualNetworkRequest(NetworkTopology):
    def __init__(self, name, start, lifetime, node_size=5, link_probability=0.5, weights=None, graph_type=0,
                 imported_graph=None):
        super(VirtualNetworkRequest, self).__init__(name, node_size, link_probability, weights, graph_type,
                                                    imported_graph)
        self.name = name
        self.start = start
        self.lifetime = lifetime

    def assign_resource(self, substrate_resource, virtual_node_request, type="temporary"):
        if substrate_resource["name"] == virtual_node_request["name"]:
            return {"name": substrate_resource["name"],
                    "attributes": substrate_resource["attributes"] + virtual_node_request["attributes"]}
        else:
            print("no such type of substrate resource")
            return 0

    def release_resource(self, substrate_resource, virtual_node_request):
        if substrate_resource["name"] == virtual_node_request["name"]:
            return {"name": substrate_resource["name"],
                    "attributes": substrate_resource["attributes"] - virtual_node_request["attributes"]}
        else:
            print("no such type of substrate resource")
            return 0

    def initialize_node_attributes(self, type="normal", costumize=None):
        attribute_list = list()
        if (type == "normal"):
            attribute_list.append({"name": "cpu_in_use", "attributes": np.random.uniform(25, 100, self.node_size)})
            bandwidth_max = np.zeros([self.node_size])
            for (u, v) in self.graph_topology.edges():
                bandwidth_max[u - 1] += self.graph_topology.get_edge_data(u, v)["weight"]
                bandwidth_max[v - 1] += self.graph_topology.get_edge_data(u, v)["weight"]
            attribute_list.append({"name": "bandwidth_in_use", "attributes": bandwidth_max})
        self.attribute_list = attribute_list
        return self.attribute_list


# aa = NetworkTopology('123')
# aa.generate_network_topology(100, 0.5)
bb = SubstrateNetwork('aaa', 100, link_probability=1, weights="normal")
bb.graph_topology = bb.generate_network_topology(nodes=200, link_probability=1, weights="normal")
# graph = bb.graph_topology
# short = bb.all_shortest_paths()
# print(graph[1][2]["weight"])
# print(short[1][1][3])
# print(np.zeros([6]) + np.zeros([6]))
# bb.initialize_node_attributes()
# print(bb.attribute_list[-1])
