import time
import pandas as pd
import pickle
from dataclasses import dataclass
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import multiprocessing
import os


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


    # ALGORITMOS DE COMUNIDADES

    def lovain_concurrent(self, weight = 'weight', resolution = 1, threshold = 1e-07, seed = None , n = 10):

        '''
        This functiosn is for execute louvain algorithm in parallel.

        Parameters
        ----------
        G : NetworkX graph
        weight : string or None, optional (default="weight")
            The name of an edge attribute that holds the numerical value
            used as a weight. If None then each edge has weight 1.
        resolution : float, optional (default=1)
            If resolution is less than 1, the algorithm favors larger communities.
            Greater than 1 favors smaller communities
        threshold : float, optional (default=0.0000001)
            Modularity gain threshold for each level. If the gain of modularity
            between 2 levels of the algorithm is less than the given threshold
            then the algorithm stops and returns the resulting communities.
        seed : list (lenght=n), random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        n :int, optional (default=10)
            Number of times to execute the algorithm.

        Returns
        -------
        list 
            A list of sets (partition of `G`). Each set represents one community and contains
            all the nodes that constitute it.
        '''

        import networkx.algorithms.community as nx_comm

        if seed:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(nx_comm.louvain_communities, [(self.G, weight, resolution, threshold, seed[i]) for i in range(n)])
        else:            
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                communities = pool.starmap(nx_comm.louvain_communities, [(self.G, weight, resolution, threshold) for _ in range(n)])

        return communities

    def lpa_wrapper(self, G, weight = 'weight', seed = 1):

        import networkx.algorithms.community as nx_comm
        return list(nx_comm.asyn_lpa_communities(G, weight, seed))

    def asyn_lpa_concurrent(self, G, weight = 'weight', seed = 1 , n = 10):
        

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            communities = pool.starmap(self.lpa_wrapper, [(G, weight, seed) for _ in range(n)])
        

        return [com for com in communities]

    def greedy_modularity_concurrent(self, G, weight=None, resolution=1, cutoff=1, best_n=None, n = 10):
        
        '''
        This functiosn is for execute greedy modularity algorithm in parallel.

        Parameters
        ----------
        G : NetworkX graph

        weight : string or None, optional (default=None)
            The name of an edge attribute that holds the numerical value used
            as a weight.  If None, then each edge has weight 1.
            The degree is the sum of the edge weights adjacent to the node.

        resolution : float, optional (default=1)
            If resolution is less than 1, modularity favors larger communities.
            Greater than 1 favors smaller communities.

        cutoff : int, optional (default=1)
            A minimum number of communities below which the merging process stops.
            The process stops at this number of communities even if modularity
            is not maximized. The goal is to let the user stop the process early.
            The process stops before the cutoff if it finds a maximum of modularity.

        best_n : int or None, optional (default=None)
            A maximum number of communities above which the merging process will
            not stop. This forces community merging to continue after modularity
            starts to decrease until `best_n` communities remain.
            If ``None``, don't force it to continue beyond a maximum.

        n :int, optional (default=10) 
            Number of times to execute the algorithm. 
        
        Returns:
            list (frozenset): A list of sets (partition of G). Each set represents one community and contains all the nodes that constitute it.
        
        '''
        
        
        import networkx.algorithms.community as nx_comm

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            communities = pool.starmap(nx_comm.greedy_modularity_communities, [(G, weight ,resolution, cutoff,best_n) for _ in range(n)])
        return communities
    
    def communities_length(self, communities):

        '''
        This function is for calculate the length of each partition in the community.
        '''

        a = [len(com) for com in communities]
        a.sort(reverse=True)

        return a
    
    def save_communities(self, communities, algorithm : str, params: list = [], seed = None):

        if algorithm == 'louvain' and params == []:
            params = [1, 1e-07, None]

        for i in range(len(communities)):
            params_name = ''.join(['_' + str(param) for param in params])
            params_name += '_seed_' + str(seed[i]) if seed else ''

            with open('./dataset/outputs/' + algorithm + '/' + algorithm + params_name + '_iter_' + str(i) , 'wb+') as f:
                pickle.dump(communities[i], f)

    def load_communities(self, algorithm : str, resolution = 1, threshold = 1e-07 , seed = 0, iter = 0):

        if algorithm == 'louvain':
            path = './dataset/outputs/' + algorithm + '/' + algorithm + '_'+ str(resolution) + '_' + str(threshold) + '_seed_' + str(seed) + '_iter_' + str(iter)
        
            with open(path, 'rb') as f:
                return pickle.load(f)

    def load_all_communities(self, algorithm : str):
        '''
        This function is for load all the communities generated by one algorithm.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to load the communities.
        
        Returns
        -------
        list : list
            A list of communities present in the folder of the algorithm.
        '''

        if algorithm == 'louvain':
            
            paths = os.listdir('./dataset/outputs/' + algorithm + '/')
        
            communities = []
            for path in paths:
                with open('./dataset/outputs/' + algorithm + '/' + path, 'rb') as f:
                    communities.append(pickle.load(f))
                    
            return communities    

def writter(lis, name):

    with open('./dataset/outputs/' + name, 'w') as f:
        for (id, value) in lis:
            f.write(f'{id}, {value}\n')  



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

    communities = m.lovain_concurrent(seed= [1,2,3,4,5,6,7,8,9,10],  n=10)

    for com in communities:
        print(m.communities_length(com))

    m.save_communities(communities, 'louvain', params=[1, 1e-07], seed= [1,2,3,4,5,6,7,8,9,10])

    # c = pickle.load(open('./dataset/outputs/louvain/louvain_1_1e-07_seed_1_iter_0', 'rb'))

    # print(m.communities_length(c))
        
    # coun_int = 0
    # coun_out = 0    
    
    
    # sorted_degree = sorted(list(m.G.degree()), key=lambda x: x[1], reverse=True)

    # for i in range(0, 9700):
    #     node = sorted_degree[i][0]
    #     if m.G.in_degree(node) >  m.G.out_degree(node):
    #         coun_int += 1
    #     else:
    #         coun_out += 1

    # print(coun_int, coun_out)


    # sorted_weighted = sorted(list(m.G.degree(weight='weight')), key=lambda x: x[1], reverse=True)

    # set_degree = set()
    
    # for i in range(0, 9700):
    #     set_degree.add(sorted_degree[i][0])

    # coun = 0

    # for i in range(0, 9700):
    #     if sorted_weighted[i][0] in set_degree:
    #         coun += 1

    # print(coun)
    
    #print(sum([x[1] for x in sorted_degree[0:3995]]))

    #de_cen = nx.degree_centrality(m.G)

    #print(de_cen)
    #print(sorted(de_cen.items(), key=lambda x: x[1]))

    #print(m.G.degree('TH-F-300002'))

    # a = m.G.degree(weight='weight')
    
    # writter(a, 'weight_degree_30.csv')
    
    # a = m.G.in_degree(weight='weight')
    
    # writter(a, 'weight_in_degree_30.csv')    

    # a = m.G.out_degree(weight='weight')
    
    # writter(a, 'weight_out_degree_30.csv')
    
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

    #comm = nx_comm.asyn_lpa_communities(m.G, weight='weight', seed=5)
     

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

    # comm = nx_comm.greedy_modularity_communities(m.G, weight=None)

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
   
    # print(len(q.get().communities))
    # print(q.get().method_name)

    # print(len(q.get().communities))
    # print(q.get().method_name)

    # for pro in procs:
    #     pro.join()

    # time2 = time.time()
    # print(time2 - time1)
    # singles_nodes = []

    
    # for q in coms.communities:       
    #     a.append(len(q))
    #     a.sort()
    #     if len(q) == 1:
    #         singles_nodes.append(q)
    # print(a)
    # a = []
    # print(singles_nodes)
    # singles_nodes = []
    
    
    
    # start = time.time()
    # b = nx_comm.louvain_communities(m.G, seed= 1)
    # end = time.time()
    # print(end - start)

    # for g in b:       
    #     a.append(len(g))
    #     a.sort()
    
    # print(a)