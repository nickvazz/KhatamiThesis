import glob

import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from networkx.algorithms.approximation import clique
# only imports if python3
import nxviz as nv

from scipy.linalg import block_diag

sns.set()

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

for U in [4,5,6,8,9,10,12,14,16,20]:
    data = glob.glob('Results/FirstSolidRun/U{}/2D*.csv'.format(U))[0]
    df = pd.read_csv(data)[['T','x','y']]

    num_blocks = {}

    Ns = range(16,33)

    heatmap = np.zeros(shape=(len(Ns),len(df['T'].unique())))
    connected = np.zeros(shape=(len(Ns),len(df['T'].unique())))
    pathlength = np.zeros(shape=(len(Ns),len(df['T'].unique())))
    radii = np.zeros(shape=(len(Ns),len(df['T'].unique())))
    total_subgraphs = np.zeros(shape=(len(Ns),len(df['T'].unique())))
    block_totals = np.zeros(shape=(len(Ns),len(df['T'].unique())))

    for idxT, T in enumerate(df['T'].unique()):
        num_blocks[T] = []
        for idxN, N in enumerate(Ns):
            print ('U:{}, T:{}, N:{}'.format(U,T,N))
            # print ('\rT={}:\t N={}'.format(T,N), end='',flush=True)

            X_temp = df[df['T'] == T][['x','y']].iloc[:100,:]

            nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(X_temp)
            distances, indices = nbrs.kneighbors(X_temp)
            # adj_matrix = nbrs.kneighbors_graph(X_temp).toarray()

            edges = {idx: nodes for idx, nodes in enumerate(indices)}
            G = nx.DiGraph(edges,weight=distances)

            adj_matrix = nx.adjacency_matrix(G).todense()

            # print (block_diag(adj_matrix))
            # print(nx.average_shortest_path_length(G))

            rows = set()
            blocks = []
            for col in range(adj_matrix.shape[1]):
                row = np.argmax(adj_matrix[:,col])
                if row not in rows:
                    rows.add(row)
                    blocks.append((row, col))

            block_totals[idxN, idxT] = len(blocks)
            num_blocks[T].append(len(blocks))

            # print (len(blocks))
            # print (T,N, nx.is_strongly_connected(G))
            # print (nx.degree_centrality(G),nx.in_degree_centrality(G),nx.out_degree_centrality(G))
            # print (np.asarray(list(nx.closeness_centrality(G).values())).mean())
            heatmap[idxN, idxT] = np.asarray(list(nx.closeness_centrality(G).values())).mean()
            connected[idxN, idxT] = nx.is_strongly_connected(G)

            path = 0
            radius = 0
            sub_graphs = 0

            for g in nx.strongly_connected_component_subgraphs(G):
                # print (len(g))
                # print (nx.average_shortest_path_length(g.to_undirected()))
                # print (nx.is_strongly_connected(g), nx.is_strongly_connected(G))
                # for pair in nx.all_pairs_dijkstra_path(g):
                #     print (pair)
                path += nx.average_shortest_path_length(g.to_undirected())
                radius += nx.radius(g)
                sub_graphs += 1
                # print ('\n')
            # print ('\n')
            pathlength[idxN, idxT] = path
            radii[idxN, idxT] = radius
            total_subgraphs[idxN, idxT] = sub_graphs
            # print ('blocls', len(blocks), sub_graphs)


    fig, ax = plt.subplots(2,3,figsize=(15,10),sharex=True, sharey=True)

    plt.suptitle('U{} Graph Results'.format(U))

    titles = ['closeness_centrality','fully connected?','ave pathlength','radii','sub_graphs','block totals']
    for idxA, a in enumerate(np.ravel(ax)):
        # print (idxA, a)
        a.set_title(titles[idxA])

    sns.heatmap(heatmap, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[0,0])
    sns.heatmap(connected, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[0,1])
    sns.heatmap(pathlength, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[0,2])
    sns.heatmap(radii, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[1,0])
    sns.heatmap(total_subgraphs, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[1,1])
    sns.heatmap(block_totals, xticklabels=df['T'].unique(), yticklabels=Ns, ax=ax[1,2])


    plt.savefig('Results/graph_results/U{}'.format(U))
    plt.clf()
# plt.show()

# print (num_blocks)
# print ()

def plot_blocks_per_neighbor():
    for key, value in num_blocks.items():
        # print (key, value)
        plt.plot(Ns, value, label=key)

    plt.legend()
    plt.show()

    # fig, ax = plt.subplots(2,1, figsize=(12,4))
    # sns.heatmap(adj_matrix, ax=ax[0])

    # arc = nv.ArcPlot(G)
    # arc.draw()

    # matrix = nv.MatrixPlot(G)
    # matrix.draw()

    # circos = nv.CircosPlot(G)
    # circos.draw()
    # plt.show()
