import itertools
import warnings
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from AppGenerator import AppGenerator
from ServerlessAppWorkflow import ServerlessAppWorkflow

warnings.filterwarnings("ignore")


class PerfOpt:
    def __init__(self, Appworkflow, generate_perf_profile=True, mem_list=None):
        self.App = Appworkflow
        self.appgenerator = AppGenerator(seed=16, type='4PL')
        if mem_list is None:
            self.mem_list = [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152,
                             1216,
                             1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176,
                             2240,
                             2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008]
        else:
            self.mem_list = mem_list
        if generate_perf_profile:
            self.generate_perf_profile()
        self.minimal_mem_configuration, self.maximal_mem_configuration, self.maximal_cost, self.minimal_avg_rt, self.minimal_cost, self.maximal_avg_rt = self.get_optimization_boundary()
        self.update_BCR()
        self.all_simple_paths = [path for path in
                                 nx.all_simple_paths(self.App.deloopedG, self.App.startPoint, self.App.endPoint)]
        self.simple_paths_num = len(self.all_simple_paths)
        self.CPcounter = 0

    # Generate performance curve for each node in the workflow
    def generate_perf_profile(self):
        node_list = [item for item in self.App.workflowG.nodes]
        node_list.remove('Start')
        node_list.remove('End')
        nx.set_node_attributes(self.App.workflowG, {}, 'perf_profile')
        for node in node_list:
            self.App.workflowG.nodes[node]['perf_profile'] = self.appgenerator.gen_rt_mem_data(node)

    # Update mem and rt attributes of each node in the workflow
    def update_mem_rt(self, G, mem_dict):
        for node in mem_dict:
            G.nodes[node]['mem'] = mem_dict[node]
            G.nodes[node]['rt'] = G.nodes[node]['perf_profile'][mem_dict[node]]

    # Update mem and rt attributes of each node in the workflow
    def update_App_workflow_mem_rt(self, App, mem_dict):
        self.update_mem_rt(App.workflowG, mem_dict)
        App.updateRT()

    def get_perf_cost_table(self, file, start_iterations=1, end_iterations=None):
        '''
        Enumerate all possible combinations of memory. For each combination, calculate the end-to-end response time and average cost.
        Save the results into a csv.
        Args:
            file (string): the name of the output csv to be saved
            start_iterations (int): the start iterations e.g. 1 == start from the first iteration, 2 == start from the second iteration
            end_iterations (int): the end iterations e.g. 10 == end after finishing the 10th iteration
        '''
        data = pd.DataFrame()
        self.App.update_NE()
        node_list = [item for item in self.App.workflowG.nodes]
        node_list.remove('Start')
        node_list.remove('End')
        all_available_mem_list = []
        for node in node_list:
            all_available_mem_list.append(
                [item for item in np.sort(list(self.App.workflowG.nodes[node]['perf_profile'].keys()))])
        if (end_iterations != None):
            task_size = end_iterations - start_iterations + 1
        else:
            task_size = np.prod([len(item) for item in all_available_mem_list]) - start_iterations + 1
        mem_configurations = itertools.product(*all_available_mem_list)
        for i in range(start_iterations - 1):
            next(mem_configurations)
        iterations_count = start_iterations - 1
        print('Get Performance Cost Table - Task Size: {}'.format(task_size))
        if (end_iterations != None):
            with tqdm(total=task_size) as pbar:
                for mem_config in mem_configurations:
                    iterations_count += 1
                    current_mem_config = dict(zip(node_list, mem_config))
                    self.update_App_workflow_mem_rt(self.App, current_mem_config)
                    current_cost = self.App.get_avg_cost()
                    self.App.get_simple_dag()
                    current_rt = self.App.get_avg_rt()
                    aRow = current_mem_config
                    aRow['Cost'] = current_cost
                    aRow['RT'] = current_rt
                    aRow = pd.Series(aRow).rename(iterations_count)
                    data = data.append(aRow)
                    pbar.update()
                    if (iterations_count >= end_iterations):
                        break
        else:
            with tqdm(total=task_size) as pbar:
                for mem_config in mem_configurations:
                    iterations_count += 1
                    current_mem_config = dict(zip(node_list, mem_config))
                    self.update_App_workflow_mem_rt(self.App, current_mem_config)
                    current_cost = self.App.get_avg_cost()
                    self.App.get_simple_dag()
                    current_rt = self.App.get_avg_rt()
                    aRow = current_mem_config
                    aRow['Cost'] = current_cost
                    aRow['RT'] = current_rt
                    aRow = pd.Series(aRow).rename(iterations_count)
                    data = data.append(aRow)
                    pbar.update()
        data.to_csv(file, index=True)

    def get_optimization_boundary(self):
        node_list = [item for item in self.App.workflowG.nodes]
        minimal_mem_configuration = {node: min(self.App.workflowG.nodes[node]['perf_profile'].keys()) for node in
                                     node_list}
        maximal_mem_configuration = {node: max(self.App.workflowG.nodes[node]['perf_profile'].keys()) for node in
                                     node_list}
        self.App.update_NE()
        self.update_App_workflow_mem_rt(self.App, maximal_mem_configuration)
        maximal_cost = self.App.get_avg_cost()
        self.App.get_simple_dag()
        minimal_avg_rt = self.App.get_avg_rt()
        self.update_App_workflow_mem_rt(self.App, minimal_mem_configuration)
        minimal_cost = self.App.get_avg_cost()
        self.App.get_simple_dag()
        maximal_avg_rt = self.App.get_avg_rt()
        return (minimal_mem_configuration, maximal_mem_configuration, maximal_cost, minimal_avg_rt, minimal_cost,
                maximal_avg_rt)

    # Get the Benefit Cost Ratio (absolute value) of each function
    def update_BCR(self):
        node_list = [item for item in self.App.workflowG.nodes]
        for node in node_list:
            available_mem_list = [item for item in np.sort(list(self.App.workflowG.nodes[node]['perf_profile'].keys()))]
            available_rt_list = [self.App.workflowG.nodes[node]['perf_profile'][item] for item in available_mem_list]
            slope, intercept = np.linalg.lstsq(np.vstack([available_mem_list, np.ones(len(available_mem_list))]).T,
                                               np.array(available_rt_list), rcond=None)[0]
            self.App.workflowG.nodes[node]['BCR'] = np.abs(slope)

    # Find the probability refined critical path in self.App
    def find_PRCP(self, order=0, leastCritical=False):
        self.CPcounter += 1
        tp_list = self.App.getTP(self.App.deloopedG, self.all_simple_paths)
        rt_list = self.App.sumRT_with_NE(self.all_simple_paths, includeStartNode=True, includeEndNode=True)
        prrt_list = np.multiply(tp_list, rt_list)
        if (leastCritical):
            PRCP = np.argsort(prrt_list)[order]
        else:
            PRCP = np.argsort(prrt_list)[-1 - order]
        return (self.all_simple_paths[PRCP])

    # Update the list of available memory configurations in ascending order
    def update_available_mem_list(self, BCR=False, BCRthreshold=0.1, BCRinverse=False):
        node_list = [item for item in self.App.workflowG.nodes]
        for node in node_list:
            if (BCR):
                available_mem_list = [item for item in
                                      np.sort(list(self.App.workflowG.nodes[node]['perf_profile'].keys()))]
                mem_zip = [item for item in zip(available_mem_list, available_mem_list[1:])]
                if (BCRinverse):
                    available_mem_list = [item for item in mem_zip if np.abs((item[1] - item[0]) / (
                            self.App.workflowG.nodes[node]['perf_profile'][item[1]] -
                            self.App.workflowG.nodes[node]['perf_profile'][item[0]])) > 1.0 / (
                                              self.App.workflowG.nodes[node]['BCR']) * BCRthreshold]
                else:
                    available_mem_list = [item for item in mem_zip if np.abs((self.App.workflowG.nodes[node][
                                                                                  'perf_profile'][item[1]] -
                                                                              self.App.workflowG.nodes[node][
                                                                                  'perf_profile'][item[0]]) / (
                                                                                     item[1] - item[0])) >
                                          self.App.workflowG.nodes[node]['BCR'] * BCRthreshold]
                available_mem_list = list(np.sort(list(set(itertools.chain(*available_mem_list)))))
            else:
                available_mem_list = [item for item in
                                      np.sort(list(self.App.workflowG.nodes[node]['perf_profile'].keys()))]
            self.App.workflowG.nodes[node]['available_mem'] = available_mem_list  # Sorted list

    def PRCPG_BPBC(self, budget, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        '''
        Probability Refined Critical Path Algorithm - Minimal end-to-end response time under a budget constraint
        Best Performance under budget constraint

        Args:
            budget (float): the budge constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'RT/M' - Benefit is RT, Cost is Mem. Eliminate mem configurations which do not conform to BCR limitations.
                                         The greedy strategy is to select the config with maximal RT reduction.
                              'ERT/C' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                             The greedy strategy is to select the config with maximal RT reduction.
                              'MAX' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                       The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'RT/M'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'ERT/C'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "RT/M"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=False)
        else:
            self.update_available_mem_list(BCR=False)
        if (BCR):
            cost = self.minimal_cost
        cost = self.minimal_cost
        surplus = budget - cost
        self.update_App_workflow_mem_rt(self.App, self.minimal_mem_configuration)
        current_avg_rt = self.maximal_avg_rt
        current_cost = self.minimal_cost
        last_e2ert_cost_BCR = 0
        order = 0
        iterations_count = 0
        while (round(surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_avg_rt_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflowG, 'mem')
            for node in cp:
                avg_rt_reduction_of_each_mem_config = {}
                for mem in reversed(self.App.workflowG.nodes[node]['available_mem']):
                    if (mem <= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, {node: mem})
                    increased_cost = self.App.get_avg_cost() - current_cost
                    if (increased_cost < surplus):
                        self.App.get_simple_dag()
                        rt_reduction = current_avg_rt - self.App.get_avg_rt()
                        if (rt_reduction > 0):
                            avg_rt_reduction_of_each_mem_config[mem] = (rt_reduction, increased_cost)
                self.update_App_workflow_mem_rt(self.App, {node: mem_backup[node]})
                if (BCR and BCRtype == "ERT/C"):
                    avg_rt_reduction_of_each_mem_config = {item: avg_rt_reduction_of_each_mem_config[item] for item in
                                                           avg_rt_reduction_of_each_mem_config.keys() if
                                                           avg_rt_reduction_of_each_mem_config[item][0] /
                                                           avg_rt_reduction_of_each_mem_config[item][
                                                               1] > last_e2ert_cost_BCR * BCRthreshold}
                if (BCR and BCRtype == "MAX"):
                    avg_rt_reduction_of_each_mem_config = {item: (
                        avg_rt_reduction_of_each_mem_config[item][0], avg_rt_reduction_of_each_mem_config[item][1],
                        avg_rt_reduction_of_each_mem_config[item][0] / avg_rt_reduction_of_each_mem_config[item][1]) for
                        item in avg_rt_reduction_of_each_mem_config.keys()}
                if (len(avg_rt_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in avg_rt_reduction_of_each_mem_config.values()])
                        max_rt_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_cost_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (reversed_dict[(
                            max_rt_reduction_under_MAX_BCR, min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                   max_rt_reduction_under_MAX_BCR,
                                                                   min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                                                                   max_BCR)
                    else:
                        max_rt_reduction = np.max([item[0] for item in avg_rt_reduction_of_each_mem_config.values()])
                        min_increased_cost_under_MAX_rt_reduction = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (
                            reversed_dict[(max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)],
                            max_rt_reduction,
                            min_increased_cost_under_MAX_rt_reduction)

            if (len(max_avg_rt_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_avg_rt_reduction_of_each_node.values()])
                max_rt_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_avg_rt_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key][3] == max_BCR and
                               max_avg_rt_reduction_of_each_node[key][1] == max_rt_reduction_under_MAX_BCR][0]
                target_mem = max_avg_rt_reduction_of_each_node[target_node][0]
            else:
                max_rt_reduction = np.max([item[1] for item in max_avg_rt_reduction_of_each_node.values()])
                min_increased_cost_under_MAX_rt_reduction = np.min(
                    [item[2] for item in max_avg_rt_reduction_of_each_node.values() if item[1] == max_rt_reduction])
                target_mem = np.min([item[0] for item in max_avg_rt_reduction_of_each_node.values() if
                                     item[1] == max_rt_reduction and item[
                                         2] == min_increased_cost_under_MAX_rt_reduction])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key] == (
                                   target_mem, max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, {target_node: target_mem})
            max_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][1]
            min_increased_cost_under_MAX_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][2]
            current_avg_rt = current_avg_rt - max_rt_reduction
            surplus = surplus - min_increased_cost_under_MAX_rt_reduction
            current_cost = self.App.get_avg_cost()
            current_e2ert_cost_BCR = max_rt_reduction / min_increased_cost_under_MAX_rt_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflowG, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        print('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        print('Average end-to-end response time: {}'.format(current_avg_rt))
        print('Average Cost: {}'.format(current_cost))
        print('PRCP_BPBC Optimization Completed.')
        return (current_avg_rt, current_cost, current_mem_configuration, iterations_count)

    def PRCPG_BCPC(self, rt_constraint, BCR=False, BCRtype="M/RT", BCRthreshold=0.1):
        '''
        Probability Refined Critical Path Algorithm - Minimal cost under an end-to-end response time constraint
        Best cost under performance (end-to-end response time) constraint

        Args:
            rt_constraint (float): End-to-end response time constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'M/RT' - Benefit is Mem, Cost is RT. (inverse) Eliminate mem configurations which do not conform to BCR limitations
                              'C/ERT' - Benefit is the cost reduction, Cost is increased ERT.
                              'MAX' - Benefit is the cost reduction, Cost is increased ERT. The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'M/RT'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'C/ERT'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "M/RT"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=True)
        else:
            self.update_available_mem_list(BCR=False)
        self.update_App_workflow_mem_rt(self.App, self.maximal_mem_configuration)
        current_avg_rt = self.minimal_avg_rt
        performance_surplus = rt_constraint - current_avg_rt
        current_cost = self.maximal_cost
        last_e2ert_cost_BCR = 0
        order = 0
        iterations_count = 0
        while (round(performance_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(leastCritical=True, order=order)
            max_cost_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflowG, 'mem')
            for node in cp:
                cost_reduction_of_each_mem_config = {}
                for mem in self.App.workflowG.nodes[node][
                    'available_mem']:
                    if (mem >= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, {node: mem})
                    self.App.get_simple_dag()
                    temp_avg_rt = self.App.get_avg_rt()
                    increased_rt = temp_avg_rt - current_avg_rt
                    cost_reduction = current_cost - self.App.get_avg_cost()
                    if (increased_rt < performance_surplus and cost_reduction > 0):
                        cost_reduction_of_each_mem_config[mem] = (cost_reduction, increased_rt)
                self.update_App_workflow_mem_rt(self.App, {node: mem_backup[node]})
                if (BCR and BCRtype == 'C/ERT'):
                    cost_reduction_of_each_mem_config = {item: cost_reduction_of_each_mem_config[item] for item in
                                                         cost_reduction_of_each_mem_config.keys() if
                                                         cost_reduction_of_each_mem_config[item][0] /
                                                         cost_reduction_of_each_mem_config[item][
                                                             1] > last_e2ert_cost_BCR * BCRthreshold}
                elif (BCR and BCRtype == "MAX"):
                    cost_reduction_of_each_mem_config = {item: (
                        cost_reduction_of_each_mem_config[item][0], cost_reduction_of_each_mem_config[item][1],
                        cost_reduction_of_each_mem_config[item][0] / cost_reduction_of_each_mem_config[item][1]) for
                        item in
                        cost_reduction_of_each_mem_config.keys()}
                if (len(cost_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in cost_reduction_of_each_mem_config.values()])
                        max_cost_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in cost_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_rt_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(cost_reduction_of_each_mem_config.values(),
                                                 cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (reversed_dict[(
                            max_cost_reduction_under_MAX_BCR, min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                 max_cost_reduction_under_MAX_BCR,
                                                                 min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                                                                 max_BCR)
                    else:
                        max_cost_reduction = np.max([item[0] for item in cost_reduction_of_each_mem_config.values()])
                        min_increased_rt_under_MAX_cost_reduction = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction])
                        reversed_dict = dict(
                            zip(cost_reduction_of_each_mem_config.values(), cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (
                            reversed_dict[(max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)],
                            max_cost_reduction,
                            min_increased_rt_under_MAX_cost_reduction)
            if (len(max_cost_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_cost_reduction_of_each_node.values()])
                max_cost_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_cost_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key][3] == max_BCR and
                               max_cost_reduction_of_each_node[key][1] == max_cost_reduction_under_MAX_BCR][0]
                target_mem = max_cost_reduction_of_each_node[target_node][0]
            else:
                max_cost_reduction = np.max([item[1] for item in max_cost_reduction_of_each_node.values()])
                min_increased_rt_under_MAX_cost_reduction = np.min(
                    [item[2] for item in max_cost_reduction_of_each_node.values() if item[1] == max_cost_reduction])
                target_mem = np.min([item[0] for item in max_cost_reduction_of_each_node.values() if
                                     item[1] == max_cost_reduction and item[
                                         2] == min_increased_rt_under_MAX_cost_reduction])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key] == (
                                   target_mem, max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, {target_node: target_mem})
            max_cost_reduction = max_cost_reduction_of_each_node[target_node][1]
            min_increased_rt_under_MAX_cost_reduction = max_cost_reduction_of_each_node[target_node][2]
            current_cost = current_cost - max_cost_reduction
            performance_surplus = performance_surplus - min_increased_rt_under_MAX_cost_reduction
            current_avg_rt = current_avg_rt + min_increased_rt_under_MAX_cost_reduction
            current_e2ert_cost_BCR = max_cost_reduction / min_increased_rt_under_MAX_cost_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflowG, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        print('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        print('Average end-to-end response time: {}'.format(current_avg_rt))
        print('Average Cost: {}'.format(current_cost))
        print('PRCPG_BCPC Optimization Completed.')
        return (current_avg_rt, current_cost, current_mem_configuration, iterations_count)

    def get_opt_curve(self, filenameprefix, budget_list, performance_constraint_list, BCRthreshold=0.2):
        '''
        Get the Optimization Curve and save as csv
        Args:
            nop_cost (int): the number of evenly spaced budgets in the range of Cost
            nop_rt (int): the number of evenly spaced performance constraints in the range of RT
        '''
        BPBC_data = pd.DataFrame()
        for budget in budget_list:
            aRow = {'Budget': budget, 'BCR_threshold': BCRthreshold}
            rt, cost, config, iterations = self.PRCPG_BPBC(budget, BCR=False)
            aRow['BCR_disabled_RT'] = rt
            aRow['BCR_disabled_Cost'] = cost
            aRow['BCR_disabled_Config'] = config
            aRow['BCR_disabled_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BPBC(budget, BCR=True, BCRtype='RT/M',
                                                           BCRthreshold=BCRthreshold)
            aRow['BCR_RT/M_RT'] = rt
            aRow['BCR_RT/M_Cost'] = cost
            aRow['BCR_RT/M_Config'] = config
            aRow['BCR_RT/M_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BPBC(budget, BCR=True, BCRtype='ERT/C',
                                                           BCRthreshold=BCRthreshold)
            aRow['BCR_ERT/C_RT'] = rt
            aRow['BCR_ERT/C_Cost'] = cost
            aRow['BCR_ERT/C_Config'] = config
            aRow['BCR_ERT/C_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BPBC(budget, BCR=True, BCRtype='MAX')
            aRow['BCR_MAX_RT'] = rt
            aRow['BCR_MAX_Cost'] = cost
            aRow['BCR_MAX_Config'] = config
            aRow['BCR_MAX_Iterations'] = iterations
            aRow = pd.Series(aRow)
            BPBC_data = BPBC_data.append(aRow, ignore_index=True)
            BPBC_data = BPBC_data[
                ['Budget', 'BCR_disabled_RT', 'BCR_RT/M_RT', 'BCR_ERT/C_RT', 'BCR_MAX_RT', 'BCR_disabled_Cost',
                 'BCR_RT/M_Cost', 'BCR_ERT/C_Cost', 'BCR_MAX_Cost', 'BCR_disabled_Config', 'BCR_RT/M_Config',
                 'BCR_ERT/C_Config', 'BCR_MAX_Config', 'BCR_disabled_Iterations', 'BCR_RT/M_Iterations',
                 'BCR_ERT/C_Iterations', 'BCR_MAX_Iterations', 'BCR_threshold']]
        BPBC_data.to_csv(filenameprefix + '_BPBC.csv', index=False)

        BCPC_data = pd.DataFrame()
        for perf_constraint in performance_constraint_list:
            aRow = {'Performance_Constraint': perf_constraint, 'BCR_threshold': BCRthreshold}
            rt, cost, config, iterations = self.PRCPG_BCPC(rt_constraint=perf_constraint, BCR=False)
            aRow['BCR_disabled_RT'] = rt
            aRow['BCR_disabled_Cost'] = cost
            aRow['BCR_disabled_Config'] = config
            aRow['BCR_disabled_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BCPC(rt_constraint=perf_constraint, BCR=True, BCRtype='RT/M',
                                                           BCRthreshold=BCRthreshold)
            aRow['BCR_M/RT_RT'] = rt
            aRow['BCR_M/RT_Cost'] = cost
            aRow['BCR_M/RT_Config'] = config
            aRow['BCR_M/RT_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BCPC(rt_constraint=perf_constraint, BCR=True,
                                                           BCRtype='ERT/C', BCRthreshold=BCRthreshold)
            aRow['BCR_C/ERT_RT'] = rt
            aRow['BCR_C/ERT_Cost'] = cost
            aRow['BCR_C/ERT_Config'] = config
            aRow['BCR_C/ERT_Iterations'] = iterations
            rt, cost, config, iterations = self.PRCPG_BCPC(rt_constraint=perf_constraint, BCR=True, BCRtype='MAX')
            aRow['BCR_MAX_RT'] = rt
            aRow['BCR_MAX_Cost'] = cost
            aRow['BCR_MAX_Config'] = config
            aRow['BCR_MAX_Iterations'] = iterations
            aRow = pd.Series(aRow)
            BCPC_data = BCPC_data.append(aRow, ignore_index=True)
            BCPC_data = BCPC_data[
                ['Performance_Constraint', 'BCR_disabled_RT', 'BCR_M/RT_RT', 'BCR_C/ERT_RT', 'BCR_MAX_RT',
                 'BCR_disabled_Cost',
                 'BCR_M/RT_Cost', 'BCR_C/ERT_Cost', 'BCR_MAX_Cost', 'BCR_disabled_Config', 'BCR_M/RT_Config',
                 'BCR_C/ERT_Config', 'BCR_MAX_Config', 'BCR_disabled_Iterations', 'BCR_M/RT_Iterations',
                 'BCR_C/ERT_Iterations', 'BCR_MAX_Iterations', 'BCR_threshold']]
        BCPC_data.to_csv(filenameprefix + '_BCPC.csv', index=False)
