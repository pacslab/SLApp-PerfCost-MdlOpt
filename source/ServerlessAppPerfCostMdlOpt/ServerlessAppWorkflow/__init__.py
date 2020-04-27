import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import warnings

warnings.filterwarnings("ignore")


class ServerlessAppWorkflow:
    def __init__(self, G, delayType='SFN', platform='AWS', pricing_model=None, PGS=None, PPI=None):
        if ('Start' in G.nodes and 'End' in G.nodes):
            G.nodes['Start']['rt'] = 0
            G.nodes['End']['rt'] = 0
            G.nodes['Start']['mem'] = 0
            G.nodes['End']['mem'] = 0
            G.nodes['Start']['perf_profile'] = {0: 0}
            G.nodes['End']['perf_profile'] = {0: 0}
        else:
            raise Exception('No Start and End points.')
        self.workflowG = G.copy()
        self.simpleDAG = G.copy()
        self.deloopedG = G.copy()
        self.startPoint = 'Start'
        self.endPoint = 'End'
        self.rt = nx.get_node_attributes(self.workflowG, 'rt')
        self.DAGrt = nx.get_node_attributes(self.simpleDAG, 'rt')
        if (delayType == 'None'):
            self.nodeDelay = {node: 0 for node in self.workflowG.nodes}
            self.edgeDelay = {edge: 0 for edge in self.workflowG.edges}
        elif (delayType == 'SFN'):
            self.nodeDelay = {node: 18.81 for node in self.workflowG.nodes}
            self.edgeDelay = {edge: 1 for edge in self.workflowG.edges}
        elif (delayType == 'Defined'):
            self.nodeDelay = nx.get_node_attributes(self.workflowG, 'delay')
            self.edgeDelay = nx.get_edge_attributes(self.workflowG, 'delay')
        self.p_node_num = 1
        self.b_node_num = 1
        self.mem = nx.get_node_attributes(self.workflowG,
                                          'mem')  # Memory configuration of each function in the workflow
        self.platform = platform

        if pricing_model is None:
            self.pricing_model = {0: 0, 128: 0.000000208, 192: 0.000000313, 256: 0.000000417, 320: 0.000000521,
                                  384: 0.000000625, 448: 0.000000729, 512: 0.000000834, 576: 0.000000938,
                                  640: 0.000001042,
                                  704: 0.000001146, 768: 0.00000125, 832: 0.000001354, 896: 0.000001459,
                                  960: 0.000001563,
                                  1024: 0.000001667, 1088: 0.000001771, 1152: 0.000001875, 1216: 0.00000198,
                                  1280: 0.000002084, 1344: 0.000002188, 1408: 0.000002292, 1472: 0.000002396,
                                  1536: 0.000002501, 1600: 0.000002605, 1664: 0.000002709, 1728: 0.000002813,
                                  1792: 0.000002917, 1856: 0.000003021, 1920: 0.000003126, 1984: 0.00000323,
                                  2048: 0.000003334, 2112: 0.000003438, 2176: 0.000003542, 2240: 0.000003647,
                                  2304: 0.000003751, 2368: 0.000003855, 2432: 0.000003959, 2496: 0.000004063,
                                  2560: 0.000004168, 2624: 0.000004272, 2688: 0.000004376, 2752: 0.00000448,
                                  2816: 0.000004584, 2880: 0.000004688, 2944: 0.000004793, 3008: 0.000004897}
        # https://aws.amazon.com/lambda/pricing/
        else:
            self.pricing_model = pricing_model
        if PGS is None:
            self.pgs = 0.0000166667
        else:
            self.pgs = PGS
        if PPI is None:
            self.ppr = 0.0000002  # Price per request
        else:
            self.ppr = PPI
        self.ne = {}
        self.approximations = []

    def updateRT(self):
        self.rt = nx.get_node_attributes(self.workflowG, 'rt')
        self.DAGrt = nx.get_node_attributes(self.simpleDAG, 'rt')

    def remove_paths(self, paths):
        if (type(paths[0]) == list):
            for path in paths:
                for i in range(1, len(path) - 1):
                    self.simpleDAG.remove_edge(path[i], path[i + 1])
                    self.simpleDAG.remove_node(path[i])
        else:
            for i in range(1, len(paths) - 1):
                self.simpleDAG.remove_edge(paths[i], paths[i + 1])
                self.simpleDAG.remove_node(paths[i])

    # Calculate the sum of RT of each path in a set of paths. (start node and end node not included by default)
    def sumRT(self, paths, includeStartNode=False, includeEndNode=False, includeFirstEdgeDelay=False,
              includeLastEdgeDelay=False):
        start = 1 - includeStartNode
        end = -1 if not includeEndNode else None
        start_edge = 1 - includeFirstEdgeDelay
        end_edge = -1 if not includeLastEdgeDelay else None
        if (type(paths[0]) == list):
            rt_results = []
            for path in paths:
                edges = [(first, second) for first, second in
                         zip(path[start_edge: end_edge], path[start_edge + 1:end_edge])]
                edge_delay = sum([self.edgeDelay[edge] for edge in edges])
                rt_results.append(
                    sum([self.DAGrt.get(name) + self.nodeDelay[name] for name in path[start:end]]) + edge_delay)
            return rt_results
        else:
            edges = [(first, second) for first, second in
                     zip(paths[start_edge: end_edge], paths[start_edge + 1:end_edge])]
            edge_delay = sum([self.edgeDelay[edge] for edge in edges])
            return sum([self.DAGrt.get(name) + self.nodeDelay[name] for name in paths[start:end]]) + edge_delay

    # Calculate the sum of RT of each path in a given graph.
    def sumRT_with_NE(self, paths, includeStartNode=False, includeEndNode=False):
        self.rt = nx.get_node_attributes(self.workflowG, 'rt')
        self.ne = nx.get_node_attributes(self.workflowG, 'ne')
        start = 1 - includeStartNode
        end = -1 if not includeEndNode else None
        if (type(paths[0]) == list):
            return list(map(lambda x: sum([self.rt.get(name) * self.ne.get(name) for name in x[start:end]]), paths))
        else:
            return sum([self.rt.get(name) * self.ne.get(name) for name in paths[start:end]])

    # Calculate the product of TP of each path in a set of paths.
    def getTP(self, G, paths):
        if (paths == []):
            return []
        if (type(paths[0]) == list):
            res = []
            for path in map(nx.utils.pairwise, paths):
                res.append(np.prod(list(map(lambda edge: G.get_edge_data(edge[0], edge[1])['weight'], list(path)))))
            return res
        else:
            return self.getTP(G, [paths])[0]

    # Get the number of executions of each function in the workflow.
    def get_avg_NE(self, G, startPoint):
        nx.set_node_attributes(G, 1, name='ne')
        ne = nx.get_node_attributes(G, 'ne')
        while [item for item in nx.simple_cycles(G)] != []:
            self_loops = [item for item in nx.selfloop_edges(G, data=True)]
            self_loops = [(item[0], item[2]['weight'], ne[item[0]]) for item in self_loops]
            for aSelfLoop in self_loops:
                out_edges = {item: G.get_edge_data(item[0], item[1])['weight'] for item in
                             G.out_edges(aSelfLoop[0])}
                tp_list = [out_edges[item] for item in out_edges]
                if (round(np.sum(tp_list), 8) != 1.0):
                    raise Exception(f'Invalid Self-loop: Node:{aSelfLoop[0]}')
                new_ne = aSelfLoop[2] / (1 - aSelfLoop[1])
                G.nodes[aSelfLoop[0]]['ne'] = new_ne
                G.remove_edge(aSelfLoop[0], aSelfLoop[0])
                del out_edges[(aSelfLoop[0], aSelfLoop[0])]
                tp_denominator = 1.0 - aSelfLoop[1]
                for edge in out_edges:
                    G[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator
            ne = nx.get_node_attributes(G, 'ne')
            cycles = [item for item in nx.simple_cycles(G)]
            try:
                cycle_by_dfs = [item[0] for item in nx.find_cycle(G)]
                cycle_by_dfs = [item for item in
                                nx.all_simple_paths(G, source=cycle_by_dfs[0], target=cycle_by_dfs[-1])]
                for item in cycle_by_dfs:
                    if not item in cycles:
                        cycles.append(item)
            except:
                pass
            cycles = [cycle for cycle in cycles if (
                    nx.shortest_path_length(G, source='Start', target=cycle[0]) < nx.shortest_path_length(G, source='Start', target=cycle[-1]))]
            cycles_dict = {}
            for item in cycles:
                if (G.has_edge(item[-1], item[0]) and (item[0], item[-1]) not in cycles_dict.keys()):
                    cycles_dict[(item[0], item[-1])] = G.get_edge_data(item[-1], item[0])['weight']
            for key in cycles_dict:
                nodes_in_cycle = list(set([item for item in itertools.chain.from_iterable(
                    [item for item in nx.all_simple_paths(G, key[0], key[1])])]))
                out_edges = {item: G.get_edge_data(item[0], item[1])['weight'] for item in G.out_edges(key[1])}
                tp_list = [out_edges[item] for item in out_edges]
                if (round(np.sum(tp_list), 4) > 1.0):
                    raise Exception(f'Invalid Loop: Node:{key[1]}')
                for node in nodes_in_cycle:
                    G.nodes[node]['ne'] = G.nodes[node]['ne'] / (1.0 - cycles_dict[key])
                G.remove_edge(key[1], key[0])
                del out_edges[(key[1], key[0])]
                tp_denominator = 1.0 - cycles_dict[key]
                for edge in out_edges:
                    G[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator
                ne = nx.get_node_attributes(G, 'ne')
        for node in G.nodes:
            paths = [item for item in nx.all_simple_paths(G, startPoint, node)]
            sum_tp = round(np.sum(self.getTP(G, paths)), 4)
            if (node == startPoint):
                sum_tp = 1
            if sum_tp < 1.0:
                G.nodes[node]['ne'] = G.nodes[node]['ne'] * sum_tp
        return (G, nx.get_node_attributes(G, 'ne'))

    def update_NE(self):
        G, ne = self.get_avg_NE(self.workflowG.copy(), self.startPoint)
        ne = {key: {'ne': ne[key]} for key in ne}
        ne['Start'] = {'ne': 0}
        ne['End'] = {'ne': 0}
        nx.set_node_attributes(self.workflowG, ne)
        self.ne = nx.get_node_attributes(self.workflowG, 'ne')
        self.deloopedG = G

    def get_avg_cost(self):
        num_exe = [item for item in self.ne.values()]
        self.mem = nx.get_node_attributes(self.workflowG, 'mem')
        if self.platform == 'AWS':
            mem_cost = [self.pricing_model[item] for item in self.mem.values()]  # Rounding PGS for AWS billing
        else:
            mem_cost = [item * self.pgs for item in self.mem.values()]
        self.rt = nx.get_node_attributes(self.workflowG, 'rt')
        billed_duration = [np.ceil(item / 100) for item in self.rt.values()]
        avg_cost = np.multiply((np.multiply(mem_cost, billed_duration) + self.ppr), num_exe)
        return np.sum(avg_cost) * 1000000

    def get_RTTP_for_paths_with_B_node(self, paths_with_B_node):
        RTTP = []
        for path in paths_with_B_node:
            B_nodes = [node for node in path if type(node) == str and node[0] == 'B']
            B_node_rt_dict = {bnode: self.DAGrt.get(bnode) for bnode in B_nodes}
            for bnode in B_nodes:
                self.DAGrt[bnode] = 0
            path_B_node_zeroing_rt = self.sumRT(paths=path, includeStartNode=False, includeEndNode=False,
                                                includeFirstEdgeDelay=True, includeLastEdgeDelay=True)
            for bnode in B_node_rt_dict.keys():
                self.DAGrt[bnode] = B_node_rt_dict[bnode]
            if (len(B_nodes) > 1):
                rt_list = [self.simpleDAG.nodes[Bnode]['rt_list'] for Bnode in B_nodes]
                tp_list = [self.simpleDAG.nodes[Bnode]['tp_list'] for Bnode in B_nodes]
                rt_list = [np.sum(item) for item in itertools.product(*rt_list)]
                tp_list = [np.prod(item) for item in list(itertools.product(*tp_list))]
            else:
                rt_list = self.simpleDAG.nodes[B_nodes[0]]['rt_list']
                tp_list = self.simpleDAG.nodes[B_nodes[0]]['tp_list']
            rt_list = [rt + path_B_node_zeroing_rt for rt in rt_list]
            RTTP.append(list(zip(rt_list, tp_list)))
        return RTTP

    def process_RTTP(self, RTTP, rt_tp1_path):
        if (rt_tp1_path != None):
            RTTP.append([(rt_tp1_path, 1)])
        TPPT_product = [item for item in itertools.product(*RTTP)]
        refined_rt = [(np.max([tup[0] for tup in item])) for item in TPPT_product]
        refined_tp = [(round(np.product([tup[1] for tup in item]), 4)) for item in TPPT_product]
        return list(zip(refined_rt, refined_tp))

    def drawGraph(self, G):
        pos = nx.planar_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        pos_higher_offset = {}
        for k, v in pos.items():
            pos_higher_offset[k] = (v[0], v[1] + 0.05)
        labels = nx.get_node_attributes(G, 'rt')
        nx.draw_networkx_labels(G, pos_higher_offset, labels=labels, label_pos=0)
        plt.show()

    def get_parallel_paths_p1(self, G):
        p1_path_list = []
        for u in G.nodes():
            for v in G.nodes():
                if u == v:
                    continue
                filtered_path = self.path_filter(G, u, v)
                paths_tp1_filter = filter(lambda p: G.get_edge_data(p[0], p[1])['weight'] == 1, filtered_path)
                paths_tp1 = []
                try:
                    for item in paths_tp1_filter:
                        paths_tp1.append(item)
                except:
                    continue
                if len(paths_tp1) > 1:
                    p1_path_list += [paths_tp1]
        return p1_path_list

    def get_parallel_paths(self, G):
        path_list = []
        for u in G.nodes():
            for v in G.nodes():
                if u == v:
                    continue
                filtered_path = self.path_filter(G, u, v)
                if len(filtered_path) > 1:
                    path_list += [filtered_path]
        return path_list

    def path_filter(self, G, u, v):
        paths = []
        for path in nx.all_simple_paths(G, u, v):
            if len(path) == 2 or max(G.degree(node) for node in path[1:-1]) == 2:
                paths += [path]
        return paths

    def isSimple(self):
        try:
            nx.find_cycle(self.simpleDAG)
            return False;
        except nx.NetworkXNoCycle:
            pass
        B_node_num = len([None for item in self.simpleDAG.nodes if type(item) == 'str' and item[0] == 'B'])
        if B_node_num != 0:
            return False
        TP_sum = 0
        paths = nx.all_simple_paths(self.simpleDAG, self.startPoint, self.endPoint)
        for path in map(nx.utils.pairwise, paths):
            TP_sum += (
                np.prod(list(map(lambda edge: self.simpleDAG.get_edge_data(edge[0], edge[1])['weight'], list(path)))))
        if (round(TP_sum, 4) == 1):
            return True
        else:
            return False

    def get_simple_dag(self):
        self.simpleDAG = self.workflowG.copy()
        self.updateRT()
        while (not self.isSimple()):
            processed = self.simplify_loops()
            if processed:
                continue
            processed = self.simplify_parallels()
            if processed:
                continue
            processed = self.simplify_branches()
            if processed:
                continue

    def get_approximations(self):
        if len(self.approximations) == 0:
            return False
        print('Models made {} approximations for {} cycles.'.format(len(self.approximations), len(
            set(apx['cycle'] for apx in self.approximations))))
        for apx in self.approximations:
            print(
                'Jumping edge from {} to {} in the cycle between {} and {} was removed, resulting in {} RT/Cost.'.format(
                    apx['jump_from'], apx['jump_to'], apx['cycle'][0], apx['cycle'][1], apx['type']))
        return True

    def get_avg_rt(self):
        paths = nx.all_simple_paths(self.simpleDAG, self.startPoint, self.endPoint)
        return np.sum([self.sumRT(item, includeStartNode=True, includeEndNode=True, includeFirstEdgeDelay=True,
                                  includeLastEdgeDelay=True) * self.getTP(self.simpleDAG, item) for item in paths])

    def merge_parallel_paths(self, parallel_paths):
        paths_tp1_sum_rt = list(map(
            lambda x: self.sumRT(paths=x, includeStartNode=False, includeEndNode=False, includeFirstEdgeDelay=True,
                                 includeLastEdgeDelay=True), parallel_paths))
        max_rt_path_index = np.argmax(paths_tp1_sum_rt)
        max_rt_tp1 = paths_tp1_sum_rt[max_rt_path_index]
        max_rt_path = parallel_paths[max_rt_path_index]
        node_name = 'P{}'.format(self.p_node_num)
        self.p_node_num += 1
        self.simpleDAG.add_node(node_name, rt=max_rt_tp1, path=max_rt_path)
        self.simpleDAG.add_weighted_edges_from([(max_rt_path[0], node_name, 1)])
        self.simpleDAG.add_weighted_edges_from([(node_name, max_rt_path[-1], 1)])
        self.nodeDelay[node_name] = 0
        self.edgeDelay[(max_rt_path[0], node_name)] = 0
        self.edgeDelay[(node_name, max_rt_path[-1])] = 0
        self.remove_paths(parallel_paths)
        self.updateRT()

    def process_self_loops(self):
        processed = False
        self_loops = [item for item in nx.selfloop_edges(self.simpleDAG, data=True)]
        self_loops = [(item[0], item[2]['weight'],
                       self.rt[item[0]] + self.nodeDelay[item[0]] + self.edgeDelay[(item[0], item[0])]) for item in
                      self_loops]
        for aSelfLoop in self_loops:
            processed = True
            out_edges = {item: self.simpleDAG.get_edge_data(item[0], item[1])['weight'] for item in
                         self.simpleDAG.out_edges(aSelfLoop[0])}
            tp_list = [out_edges[item] for item in out_edges]
            if (round(np.sum(tp_list), 8) != 1.0):
                raise Exception(f'Invalid Self-loop: Node:{aSelfLoop[0]}')
            new_rt = aSelfLoop[2] / (1 - aSelfLoop[1])
            self.simpleDAG.nodes[aSelfLoop[0]]['rt'] = new_rt
            self.simpleDAG.remove_edge(aSelfLoop[0], aSelfLoop[0])
            del out_edges[(aSelfLoop[0], aSelfLoop[0])]
            tp_denominator = 1.0 - aSelfLoop[1]
            for edge in out_edges:
                self.simpleDAG[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator
        self.updateRT()
        return processed

    def simplify_parallels(self):
        processed = False
        parallel_paths = self.get_parallel_paths_p1(self.simpleDAG)
        for pp in parallel_paths:
            paths_with_B_node = [p for p in pp if
                                 len([node for node in p if type(node) == str and node[0] == 'B']) != 0]
            if (len(paths_with_B_node) != 0):
                pp_tp_1 = pp.copy()
                for path in paths_with_B_node:
                    pp_tp_1.remove(path)
                if len(pp_tp_1) > 1:
                    self.merge_parallel_paths(pp_tp_1)
                    processed = True
                    continue
                if (len(pp_tp_1) != 0):
                    pp_tp_1_rt = self.sumRT(paths=pp_tp_1[0], includeStartNode=False, includeEndNode=False,
                                            includeFirstEdgeDelay=True, includeLastEdgeDelay=True)
                else:
                    pp_tp_1_rt = None
                RTTP = self.get_RTTP_for_paths_with_B_node(paths_with_B_node)
                refined_RTTP = self.process_RTTP(RTTP, pp_tp_1_rt)
                self.remove_paths(paths_with_B_node)
                if (len(pp_tp_1) != 0):
                    self.remove_paths(pp_tp_1[0])
                for PNode in refined_RTTP:
                    node_name = 'P{}'.format(self.p_node_num)
                    self.p_node_num += 1
                    self.simpleDAG.add_node(node_name, rt=PNode[0])
                    self.simpleDAG.add_weighted_edges_from([(pp[0][0], node_name, PNode[1]), (node_name, pp[0][-1], 1)])
                    self.nodeDelay[node_name] = 0
                    self.edgeDelay[(pp[0][0], node_name)] = 0
                    self.edgeDelay[(node_name, pp[0][-1])] = 0
                    processed = True
                continue
            self.merge_parallel_paths(pp)
            processed = True
        self.updateRT()
        return processed

    def simplify_branches(self):
        processed = False
        if (self.isSimple()):
            return
        parallel_paths = self.get_parallel_paths(self.simpleDAG)
        for pp in parallel_paths:
            out_degree_pr = sum(map(lambda p: self.simpleDAG.get_edge_data(p[0], p[1])['weight'], pp))
            if (out_degree_pr == 1):
                paths_sum_rt = self.sumRT(paths=pp, includeStartNode=False, includeEndNode=False,
                                          includeFirstEdgeDelay=True, includeLastEdgeDelay=True)
                paths_tp = self.getTP(self.simpleDAG, pp)
                node_name = 'B{}'.format(self.b_node_num)
                self.b_node_num += 1
                self.simpleDAG.add_node(node_name, rt_list=paths_sum_rt, tp_list=paths_tp, original_paths=pp,
                                        rt=np.sum(np.multiply(paths_sum_rt, paths_tp)))
                self.simpleDAG.add_weighted_edges_from([(pp[0][0], node_name, 1), (node_name, pp[0][-1], 1)])
                self.nodeDelay[node_name] = 0
                self.edgeDelay[(pp[0][0], node_name)] = 0
                self.edgeDelay[(node_name, pp[0][-1])] = 0
                self.remove_paths(pp)
                processed = True
                continue
            path_tp1_filter = filter(lambda p: self.simpleDAG.get_edge_data(p[0], p[1])['weight'] == 1, pp)
            try:
                path_tp1 = next(path_tp1_filter)
                path_tp1_rt = self.sumRT(paths=path_tp1, includeStartNode=False, includeEndNode=False,
                                         includeFirstEdgeDelay=True, includeLastEdgeDelay=True)
            except:
                continue
            pp.remove(path_tp1)
            paths_sum_rt = list(map(
                lambda x: self.sumRT(paths=x, includeStartNode=False, includeEndNode=False, includeFirstEdgeDelay=True,
                                     includeLastEdgeDelay=True), pp))
            paths_gte_path_tp1_rt_filter = filter(lambda x: paths_sum_rt[x] >= path_tp1_rt, range(0, len(pp)))
            paths_gte_path_tp1_rt_index = list(paths_gte_path_tp1_rt_filter)
            paths_tpn1_prob_sum = sum(
                [self.simpleDAG.get_edge_data(pp[i][0], pp[i][1])['weight'] for i in paths_gte_path_tp1_rt_index])
            if (paths_tpn1_prob_sum > 1):
                raise Exception('The sum of probabilities is greater than 1. Paths:{}'.format(pp))
            self.simpleDAG[path_tp1[0]][path_tp1[1]]['weight'] = 1.0 - paths_tpn1_prob_sum
            processed = True
            paths_lt_path_tp1_rt_index = list(set(range(0, len(pp))) - set(paths_gte_path_tp1_rt_index))
            for index in paths_lt_path_tp1_rt_index:
                self.remove_paths(pp[index])
        self.updateRT()
        return processed

    def simplify_loops(self):
        processed = False
        self.process_self_loops()
        cycles = [item for item in nx.simple_cycles(self.simpleDAG)]
        try:
            cycle_by_dfs = [item[0] for item in nx.find_cycle(self.simpleDAG)]
            cycle_by_dfs = [item for item in
                            nx.all_simple_paths(self.simpleDAG, source=cycle_by_dfs[0], target=cycle_by_dfs[-1])]
            for item in cycle_by_dfs:
                if not item in cycles:
                    cycles.append(item)
        except:
            pass
        cycles = [cycle for cycle in cycles if (
                nx.shortest_path_length(self.simpleDAG, source='Start', target=cycle[0]) < nx.shortest_path_length(
            self.simpleDAG, source='Start', target=cycle[-1]))]
        cycles_rttp = [((item[0], item[-1]), ((self.sumRT(item, includeStartNode=True, includeEndNode=False,
                                                          includeFirstEdgeDelay=False, includeLastEdgeDelay=True) +
                                               self.edgeDelay[(item[-1], item[0])]), self.getTP(self.simpleDAG, item)))
                       for item in cycles]
        cycles_dict = {}
        for (key, rttp) in cycles_rttp:
            if (rttp[0] != False and cycles_dict.setdefault(key, {'tp_sum': 0, 'avg_rt': 0}) != False):
                cycles_dict[key]['tp_sum'] += rttp[1]
                if (round(cycles_dict[key]['tp_sum'], 4) != 1):
                    nodes_in_cycle = list(
                        set([node for path in nx.all_simple_paths(self.simpleDAG, key[0], key[1]) for node in path]))
                    for node_in_cycle in nodes_in_cycle:
                        if node_in_cycle == key[0] or node_in_cycle == key[1]:
                            continue
                        out_nodes = set([edge[1] for edge in self.simpleDAG.out_edges(node_in_cycle)])
                        if out_nodes & set(nodes_in_cycle) != out_nodes:
                            jump_nodes = list(out_nodes - (out_nodes & set(nodes_in_cycle)))
                            for jumpnode in jump_nodes:
                                out_edges = {item: self.simpleDAG.get_edge_data(item[0], item[1])['weight'] for item in
                                             self.simpleDAG.out_edges(node_in_cycle)}
                                tp_list = [out_edges[item] for item in out_edges]
                                if (round(np.sum(tp_list), 4) > 1.0):
                                    raise Exception(f'Invalid Node: Node:{key[1]}')
                                tp_denominator = 1.0 - self.simpleDAG.get_edge_data(node_in_cycle, jumpnode)['weight']
                                self.simpleDAG.remove_edge(node_in_cycle, jumpnode)
                                del out_edges[(node_in_cycle, jumpnode)]
                                for edge in out_edges:
                                    self.simpleDAG[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator
                                if nx.shortest_path_length(self.simpleDAG, source='Start',
                                                           target=jumpnode) > nx.shortest_path_length(self.simpleDAG,
                                                                                                      source='Start',
                                                                                                      target=key[1]):
                                    self.approximations.append(
                                        {'cycle': key, 'type': 'overestimated', 'jump_from': node_in_cycle,
                                         'jump_to': jumpnode, 'tp': 1 - tp_denominator})
                                elif nx.shortest_path_length(self.simpleDAG, source='Start',
                                                             target=jumpnode) < nx.shortest_path_length(self.simpleDAG,
                                                                                                        source='Start',
                                                                                                        target=key[1]):
                                    self.approximations.append(
                                        {'cycle': key, 'type': 'underestimated', 'jump_from': node_in_cycle,
                                         'jump_to': jumpnode, 'tp': 1 - tp_denominator})
                cycles_dict[key]['avg_rt'] += rttp[0] * rttp[1]
            else:
                cycles_dict[key] = False
        cycles_dict = {key: cycles_dict[key]['avg_rt'] for key in cycles_dict if
                       cycles_dict[key] != False and round(cycles_dict[key]['tp_sum'], 4) == 1}
        for key in cycles_dict:
            loop_tp = self.simpleDAG.get_edge_data(key[1], key[0])['weight']
            out_edges = {item: self.simpleDAG.get_edge_data(item[0], item[1])['weight'] for item in
                         self.simpleDAG.out_edges(key[1])}
            tp_list = [out_edges[item] for item in out_edges]
            if (round(np.sum(tp_list), 4) > 1.0):
                raise Exception(f'Invalid Loop: Node:{key[1]}')
            new_rt = (self.DAGrt[key[1]] + loop_tp * cycles_dict[key]) / (1 - loop_tp)
            self.simpleDAG.nodes[key[1]]['rt'] = new_rt
            self.simpleDAG.remove_edge(key[1], key[0])
            processed = True
            del out_edges[(key[1], key[0])]
            tp_denominator = 1.0 - loop_tp
            for edge in out_edges:
                self.simpleDAG[edge[0]][edge[1]]['weight'] = out_edges[edge] / tp_denominator
        self.updateRT()
        return processed
