import time
import pandas as pd
import pickle
from dataclasses import dataclass
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import multiprocessing as mp


@dataclass
class Matrix:
    """    
    Class to create a matrix from a csv file and export it to a graphml file    
    
    Attributes:
        list_nodes (list): list of nodes
        pos_code_nodes (dict): dictionary with the position of each node
        G (nx.DiGraph): graph
        ady_list (list): list of adyacents nodes
        G_small (nx.DiGraph): small graph
    """

    list_nodes: list
    pos_code_nodes: dict
    G = nx.DiGraph()
    ady_list : list 
    G_small = nx.DiGraph()
    
    def __post_init__(self):        
        self.list_nodes = pickle.load(open('./dataset/nodes.pkl', 'rb'))
        self.pos_code_nodes = pickle.load(open('./dataset/pos_code_nodes.pkl', 'rb'))
        self.ady_list = [[] for _ in range(len(self.list_nodes))]
        

    def insert_nodes(self):
        self.G.add_nodes_from(self.list_nodes)
    
    def insert_weighted_edges(self):
        for i in range(len(self.ady_list)):
            for j in range(len(self.ady_list[i])):
                y = self.ady_list[i][j][0]
                weight = self.ady_list[i][j][1]
                self.G.add_edge(self.list_nodes[y], self.list_nodes[i], weight=weight)
    
    def export_graph_to_csv(self):
        nx.write_edgelist(self.G, "./dataset/30set.csv", delimiter=",", data=['weight'])

    def export_graph_to_csv_size(self, size):

        G = nx.DiGraph()
        G.add_nodes_from(self.list_nodes[:size])

        df = pd.read_csv('./dataset/matrix/0 file.csv')
        

        ady_list = [[] for _ in range(size)]

        for i in range(len(self.list_nodes[:size])):
            row = df.loc[i].to_list()
            for j in range(size):
                if row[j] != 0:
                    ady_list[i].append((j, row[j]))

        for i in range(len(ady_list)):
            for j in range(len(ady_list[i])):
                y = ady_list[i][j][0]
                weight = ady_list[i][j][1]
                G.add_edge(self.list_nodes[y], self.list_nodes[i], weight=weight)

        self.G_small = G

        nx.write_edgelist(G, f"0set_size{str(size)}.csv", delimiter=",", data=['weight'])
        nx.write_gexf(G, f"0set_size{str(size)}.gexf")
    
    def export_graph_to_graphml(self, path = "./dataset/graph_19k_3.5m.gml"):
        nx.write_graphml(self.G, path)

    def export_graph_to_adjlist(self, path = "./dataset/graph_19k_3.5m.adyl"):
        nx.write_adjlist(self.G, path)

    def load_ady_matrix(self, count = 0):

        for x in range(count+1):
            print(f'Loading: {x} file.csv')
            df = pd.read_csv(f'./dataset/matrix/{x} file.csv')            
            for i in range(df.shape[0]):
                row = df.loc[i].to_list()
                for j in range(len(row)):
                    if row[j] != 0:
                        self.ady_list[i + x * 642].append((j, row[j]))
            print(f'Finished: {x} file.csv')


        with open(f'./dataset/adym_{count}.pkl', 'wb') as f:
            pickle.dump(self.ady_list, f)
   
    def sava_matrix_obj(self, path = './dataset/graph_19k_3.5m.pkl'):
        
        with open(path, 'wb') as f:
            pickle.dump(self.G, f)

    def load_matrix_obj(self, path = './dataset/graph_19k_3.5m.pkl'):
        self.G = pickle.load(open(path, 'rb'))
    
    def read_adym(self, path = './dataset/adym_30.pkl'):
        self.ady_list = pickle.load(open(path, 'rb'))
        

if __name__ == '__main__':
    
    m = Matrix([], {},[])
    #m.load_matrix_obj()
    #m.export_graph_to_adjlist()
    m.insert_nodes()
    #m.read_adym(path='data/adym_0.pkl')
    m.read_adym(path='dataset/adym_30.pkl')
    #m.load_ady_matrix(30)    
    m.insert_weighted_edges()
    # m.sava_matrix_obj()
    print(m.G.number_of_edges())


    #de_cen = nx.degree_centrality(m.G)

    #print(de_cen)
    #print(sorted(de_cen.items(), key=lambda x: x[1]))

    #print(m.G.degree('TH-F-300002'))

    #a = m.G.degree(weight='weight')

    #print(sorted(list(a), key=lambda x: x[1], reverse=True))

    #print(nx.betweenness_centrality(m.G))

    #print(sorted(a., key=lambda x: x[1]))
    #hist = nx.degree_histogram(m.G)
    #print(hist)

    # co = 0
    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         co += hist[i]*i
    # print(co)

    # list_hist = []

    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         list_hist.append(i)

    # plt.hist(list_hist)
    # plt.savefig('hist.png')

    # var = []

    # for i in range(len(list_hist)):
    #     if list_hist[i] >= 1500:
    #         var.append(i)

    # list_hist = []

    # for i in range(len(hist)):
    #     if hist[i] != 0:
    #         if hist[i] == 1:
    #             list_hist.append(i)
    #         else:
    #             list_hist.append(i * hist[i] / 2)
    #     else:
    #         list_hist.append(0)
    
    # print(sum(list_hist))
    # print(len(var))
    # print(sum(var) / 2)
    #print(nx.function.degree_histogram(m.G))

    #ouliers = [['E0585-F-300022'], ['Gad1-F-300464'], ['VGlut-F-100031'], ['VGlut-F-100283'], ['VGlut-F-400694']]

    # for node in ouliers:
    #     id = m.pos_code_nodes[node[0]]
    #     row = m.ady_list[id]
    #     value = 0
    #     suma = 0
    #     for x in row:
    #         value += x[1]
    #     print(f'Node: {node} sum values: {value} : len : {len(m.ady_list[id])}' )
    #     print(len(m.G.adj[node[0]].items()))
    a = []
    
    time1 = time.time()

    # comm = nx_comm.asyn_lpa_communities(m.G, weight='weight')
    
    

    # for x in comm:       
    #     a.append(len(x))
    #     a.sort()
    # print(a)
    # time2 = time.time()
    # print(time2 - time1)
    
    # comm = nx_comm.greedy_modularity_communities(m.G, weight='weight')

    # time2 = time.time()
    # print(time2 - time1)

    # for x in comm:       
    #     a.append(len(x))
    #     a.sort()
    # print(a)

    #m.export_graph_to_csv()
    # m.load_ady_matrix(30)
    # for node in m.list_nodes:
    #     count += 1
    #     if count < 10:
    #         print(f'{node} : {m.G.degree[node]}')
    #         print(f'{node} : {m.G.adj[node]}')
    #     else:
    #         break
    #nx.write_gexf(m.G, "0set.gexf")
    
    #m.export_graph_to_csv_size(500)
    #print(m.G.number_of_edges())
    # print(len(list(m.G.neighbors('104198-F-000000'))))

    # print(len(list(nx.strongly_connected_components(m.G))))

    from cdlib import algorithms

    with mp.Pool(processes=mp.cpu_count()) as pool:
        output = pool.map(algorithms.infomap, [m.G for _ in range(3)])
    
    #coms = algorithms.infomap(m.G)
    
    time2 = time.time()
    print(time2 - time1)

    singles_nodes = []

    for coms in output:
        for q in coms.communities:       
            a.append(len(q))
            a.sort()
            if len(q) == 1:
                singles_nodes.append(q)
        print(a)
        a = []
        print(singles_nodes)
        singles_nodes = []
    
    
    
    # start = time.time()
    # b = nx_comm.louvain_communities(m.G)
    # end = time.time()
    # print(end - start)

    # for g in b:       
    #     a.append(len(g))
    #     a.sort()
    
    # print(a)