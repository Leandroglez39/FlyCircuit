import pandas as pd
import pickle
from dataclasses import dataclass
import networkx as nx


@dataclass
class Matrix:

    list_nodes: list
    pos_code_nodes: dict
    G = nx.DiGraph()
    ady_list : list 
    
    def __post_init__(self):
        
        self.list_nodes = pickle.load(open('./data/nodes.pkl', 'rb'))
        self.pos_code_nodes = pickle.load(open('./data/pos_code_nodes.pkl', 'rb'))
        self.ady_list = [[] for _ in range(len(self.list_nodes))]
        self.read_adym

    def insert_nodes(self):
        self.G.add_nodes_from(self.list_nodes)
    
    def insert_weighted_edges(self):
        for i in range(len(self.ady_list)):
            for j in range(len(self.ady_list[i])):
                y = self.ady_list[i][j][0]
                weight = self.ady_list[i][j][1]
                self.G.add_edge(self.list_nodes[i], self.list_nodes[y], weight=weight)

    @property
    def read_adym(self):
        self.ady_list = pickle.load(open('./data/adym_0.pkl', 'rb'))
        

if __name__ == '__main__':
    m = Matrix([], {},[])
    m.insert_nodes()
    m.insert_weighted_edges()
    for node in m.list_nodes:
        print(f'{node} : {m.G.degree[node]}')