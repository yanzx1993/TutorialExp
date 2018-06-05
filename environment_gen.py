import numpy as np
import network_topology_gen as nt
import heapq as hp
import tensorflow as tf


class Environment(object):
    def __init__(self, name, substrate_node_size, time_to_live, link_embedding_type="shortest",imported_network=None):
        self.name = name
        #self.active_VNR_list = hp.heapify
        self.time_to_live = time_to_live
        self.time = 0
        self.substrate_node_size=substrate_node_size
        if imported_network is None:
            self.substrate_network = nt.SubstrateNetwork(name + "_SubstrateNetwork", substrate_node_size, weights="normal")
        else:
            self.substrate_network=imported_network
        self.VNR_generator = VNRGenerator(name + "_VNRGenerator")
        self.link_embedding_type = link_embedding_type

        self.pending_VNR_list = list()
        self.pending_node_list = list()
        self.current_assigned_cpu = list()
        self.current_assigned_bandwidth = list()

        self.VNR_counter = 0
        self.success_embeddings = 0

        self.LARGE_PENALTY = -10000

        self.total_cost=0
        self.total_reward=0
        self.total_cost_backup=0
        self.total_reward_backup=0

        self.substrate_network_backup = self.substrate_network
        self.snode_state_backup = self.substrate_network.attribute_list

        self.current_VNR = self.fetch_new_VNR()
        self.snode_state, self.vnode_state = self.get_state()



    def get_state(self):
        snode_state = self.substrate_network.attribute_list
        snode_list=list()
        for i in range(len(snode_state)):
            snode_list.append(snode_state[i]["attributes"])
        snode_features=np.stack(snode_list)
        index = self.pending_node_list[0][0]
        vnode_feature_size = len(self.current_VNR.attribute_list)
        vnode_state = np.zeros([vnode_feature_size+1])
        for i in range(vnode_feature_size):
            vnode_state[i] = self.current_VNR.attribute_list[i]['attributes'][index]
        vnode_state[vnode_feature_size]=len(self.pending_node_list)
        return snode_features, vnode_state

    def generate_VNR(self, lifetime=1000, node_size=5, link_probability=0.5, weights="normal", graph_type=0,
                     imported_graph=None):
        start = self.time
        if self.VNR_generator is None:
            print("VNR Generator has not been initialized")
            return 0
        else:
            VNR = self.VNR_generator.generate_VNR(self.name + "_VNR", start, lifetime, node_size, link_probability,
                                                  weights, graph_type, imported_graph)
            self.pending_VNR_list.append(VNR)
            self.VNR_counter += 1
            return VNR

    def fetch_new_VNR(self):
        if len(self.pending_VNR_list) == 0:
            current_VNR = self.VNR_generator.generate_VNR(self.name + "_VNR", start=self.time, lifetime=1000)
            self.pending_VNR_list.append(current_VNR)
            # print("no VNRs to embed")
            # return 0
        # else:
        self.current_VNR = self.pending_VNR_list.pop(0)
        self.pending_node_list = self.current_VNR.node_rank("cpu_in_use")
        self.substrate_network.copy_substrate_network()
        return self.current_VNR

    def clock(self):
        self.time += 1

    def backup_env(self):
        self.substrate_network_backup = self.substrate_network
        self.total_cost_backup=self.total_cost
        self.total_reward_backup=self.total_reward

    def restore_env(self):
        self.substrate_network = self.substrate_network_backup
        self.total_cost_backup=self.total_cost
        self.total_reward_backup=self.total_reward

    def clear_partial_embedding(self):
        self.current_VNR = self.fetch_new_VNR()
        self.current_assigned_bandwidth = list()
        self.current_assigned_cpu = list()
        self.substrate_network.temporary_embedding_result = np.zeros([self.substrate_network.node_size])

    def perform_action(self, action):
        """
        
        :param action: 
        :return: 
        """
        reward = self.LARGE_PENALTY
        """link_request = self.current_VNR.graph_topology[pending_node][temporary_embedding_result[i]]
        for j in range(len(shortest_path)-1):
            link_capacity=self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j+1]]["weight"]
            if link_request>link_capacity:
                return False"""
        pending_node = self.pending_node_list.pop(0)
        node_availability = self.check_node_embedding_availability(action)
        if (node_availability == False):
            self.clear_partial_embedding()
            self.restore_env()
            return 1, reward
        temporary_embedding_result = self.substrate_network.temporary_embedding_result
        for i in range(len(temporary_embedding_result)):
            if temporary_embedding_result[i] != 0:
                if self.link_embedding_type == "shortest":
                    shortest_path = self.substrate_network.shortest_paths[action][i]
                    res = self.check_link_embedding_availability(shortest_path, pending_node,
                                                                 temporary_embedding_result[i])
                    if (res == False):
                        self.clear_partial_embedding()
                        self.restore_env()
                        return 1, reward
                    else:
                        reward=res
                else:
                    shortest_path = self.substrate_network.simple_paths(action, i)
                    success = 0
                    for l in range(len(shortest_path)):
                        res = self.check_link_embedding_availability(shortest_path[i], pending_node,
                                                                     temporary_embedding_result[i])
                        if (res != False):
                            success = 1
                            reward=res
                            break
                    if (success == 0):
                        self.clear_partial_embedding()
                        self.restore_env()
                        return 1, reward
        self.substrate_network.temporary_embedding_result[action] = pending_node
        self.substrate_network.attribute_list.get("current_embedding")[action]=1
        if (len(self.pending_node_list) == 0):
            self.success_embeddings += 1
            self.assign_resource()
            self.clear_partial_embedding()
            return 1, reward
        else:
            return 0, reward

    def check_node_embedding_availability(self, action):
        max = None
        in_use = None
        node_request = self.vnode_state[0]
        for i in range(len(self.substrate_network.attribute_list)):
            if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                in_use = self.substrate_network.attribute_list[i]["attributes"]
            if (self.substrate_network.attribute_list[i]["name"] == "cpu_max"):
                in_use = self.substrate_network.attribute_list[i]["attributes"]
        if (max[action] - in_use[action] < node_request):
            return False
        else:
            for i in range(len(self.substrate_network.attribute_list)):
                if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
                    self.substrate_network.attribute_list[i]["attributes"][action] += node_request
                self.current_assigned_cpu.append([action,node_request])
            return True

    def check_link_embedding_availability(self, shortest_path, virtual_source_node, virtual_target_node):
        """
        reward definition to be optimized
        :param shortest_path: 
        :param virtual_source_node: 
        :param virtual_target_node: 
        :return: 
        """
        cost = 0
        reward = 0
        link_request = self.current_VNR.graph_topology[virtual_source_node][virtual_target_node]["weight"]
        for j in range(len(shortest_path) - 1):
            link_capacity = \
                self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j + 1]][
                    "weight"]
            if link_request > link_capacity:
                return False
            reward+=link_request
        for j in range(len(shortest_path) - 1):
            cost+=link_request
            link_capacity = \
                self.substrate_network.graph_topology[shortest_path[j]][shortest_path[j + 1]][
                    "weight"]
            new_link_weight = link_capacity - link_request
            self.current_assigned_bandwidth[shortest_path[j]] += link_request
            self.current_assigned_bandwidth[shortest_path[j + 1]] += link_request
            self.substrate_network.graph_topology.add_edge(shortest_path[j],
                                                           [shortest_path[j + 1]],
                                                           weight=new_link_weight)
            self.current_assigned_bandwidth.append([shortest_path[j],shortest_path[j+1],link_request])
        return reward/cost

    def assign_resource(self):
        # for i in range(len(self.substrate_network.temporary_embedding_result)):
        #     if (self.substrate_network.temporary_embedding_result[i] != 0):
        #         index = self.substrate_network.temporary_embedding_result[i]
        #         self.current_assigned_cpu[i] = self.current_VNR.attribute_list[0]["attributes"][index]
        # assigned_resource = {"cpu_in_use": self.current_assigned_cpu,
        #                      "bandwidth_in_use": self.current_assigned_bandwidth,
        #                      "expire_time": self.time + self.current_VNR.lifetime}
        # for i in range(len(self.substrate_network.attribute_list)):
        #     if (self.substrate_network.attribute_list[i]["name"] == "cpu_in_use"):
        #         self.substrate_network.attribute_list[i]["attributes"] += self.current_assigned_cpu
        #     if (self.substrate_network.attribute_list[i]["name"] == "cpu_remaining"):
        #         self.substrate_network.attribute_list[i]["attributes"] -= self.current_assigned_cpu
        #     if (self.substrate_network.attribute_list[i]["name"] == "bandwidth_in_use"):
        #         self.substrate_network.attribute_list[i]["attributes"] += self.current_assigned_bandwidth
        # '''add the assigned vnr to the heap using tuples'''
        return 0

    def release_resource(self):
        """getting resource with the earliest expiring VNR and release them on substrate network based on the heap tuples"""
        return 0

    # def choose_action(self,action_prob,phase="Training"):
    #     a=np.arange(0,self.substrate_node_size)
    #     if(phase=="Training"):
    #         return np.random.choice(a, p=action_prob.eval(session=self.))
    #     else:
    #         return np.max(action_prob).__index__()

    def reward(self, action_result):
        """
        reward calculation in the environment
        possible reward function:
        
        one step or a whole trajectory?
        :return: 
        """

        return 0

    """
    stack trajectory(s,a,r,s')
    apply them to networks
    using td (critic) and policy gradient (actor)
    
    if all stacked are successful, update
    if failed...?
    1.update all using a huge negative reward
    2.update step-by-step, only affect the last failed move
    """


class VNRGenerator(object):
    def __init__(self, name, type="poisson", rate=1):
        self.name = name
        self.type = type
        self.rate = rate

    def generate_VNR(self, name, start, lifetime, node_size=5, link_probability=0.5, weights="normal", graph_type=0,
                     imported_graph=None):
        VNR = nt.VirtualNetworkRequest(name, start, lifetime, node_size, link_probability, weights, graph_type,
                                       imported_graph)
        return VNR

    def auto_generate_VNR(self, time=1000, rate=1):
        return 0

aa=Environment("env",50,10000)
prob=np.random.uniform(1,2,50)
# bb=tf.nn.softmax(prob)
# ss=aa.choose_action(bb)
# gg=aa.choose_action(bb,phase="Testing")
# print(bb)
# print(ss)
# print(gg)

