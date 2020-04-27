import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class AppGenerator:
    def __init__(self, seed=16, type='mixed', mem_list=None):
        self.seed = seed
        if mem_list is None:
            self.mem_list = [128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152,
                             1216,
                             1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176,
                             2240,
                             2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008]
        else:
            self.mem_list = mem_list
        np.random.seed(self.seed)
        if (type == 'mixed'):
            self.model_type = np.random.randint(1, 3, 100)
        elif (type == 'linear'):
            self.model_type = np.full(100, 1)
        elif (type == '4PL'):
            self.model_type = np.full(100, 2)
        self.m1_a = np.random.randint(1000, 2000, 100)
        self.m1_b = np.random.rand(100) * 2 + 0.1
        self.m1_c = np.random.randint(50, 500, 100)
        self.m1_d = (np.random.rand(100) * 0.5 + 0.01) * self.m1_a
        self.m2_k = np.random.rand(100) - 1
        self.m2_b = np.abs(3000 * (self.m2_k)) + np.random.randint(100, 1001, 100)

    def fourPL(self, x, A, B, C, D):
        return ((A - D) / (1.0 + ((x / C) ** (B))) + D)

    def ln(self, x, A, B):
        return A * x + B

    def draw_fourPL(self, xdata, ydata, params_4PL):
        x_min, x_max = np.amin(xdata), np.amax(xdata)
        xs = np.linspace(x_min, x_max, 1000)
        plt.scatter(xdata, ydata)
        plt.plot(xs, self.fourPL(xs, *params_4PL))
        plt.show()

    def draw_ln(self, xdata, ydata, params_ln):
        x_min, x_max = np.amin(xdata), np.amax(xdata)
        xs = np.linspace(x_min, x_max, 1000)
        plt.scatter(xdata, ydata)
        plt.plot(xs, self.ln(xs, *params_ln))
        plt.show()

    def gen_rt_mem_data(self, n):
        if (self.model_type[n] == 2):
            ydata = self.fourPL(np.array(self.mem_list),
                                *np.array([self.m1_a[n], self.m1_b[n], self.m1_c[n], self.m1_d[n]]))
        elif (self.model_type[n] == 1):
            ydata = self.ln(np.array(self.mem_list), self.m2_k[n], self.m2_b[n])
        return (dict(zip(self.mem_list, ydata)))

    def cycles_workflow_generator(self, node_num):
        G = nx.DiGraph()
        G.add_node('Start', pos=(0, 1))
        G.add_node('End', pos=(node_num + 1, 1))
        for i in range(node_num):
            G.add_node(i + 1, pos=(i, 1), mem=128, rt=100)
        G.add_edge('Start', 1, weight=1)
        for i in range(1, node_num + 1):
            for j in range(1, i + 1):
                G.add_edge(i, j, weight=np.around(1 / (i + 1) * 0.9, 4))
            if (i != node_num):
                G.add_edge(i, i + 1, weight=np.around(1 - np.around(np.around(1 / (i + 1) * 0.9, 4) * i, 4), 6))
        G.add_edge(node_num, 'End', weight=np.around(1 - np.around(np.around(1 / (i + 1) * 0.9, 4) * i, 4), 6))
        return G
