# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# V2: Using Undirected Graphs
import networkx as nx
import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
from scipy import mgrid
import time
from networkx.algorithms.shortest_paths import all_pairs_shortest_path
import sys


# %%
'''
if sys.argv[0].split('\\')[-1] != 'ipykernel_launcher.py':
    if len(sys.argv) < 7:
        print("Wrong format.")
        print("Try 'python SNAProject.py <N> <Pgd> <Pcn> <Pjc> <Paa> <Ppa> [<Data Percentage>]'")
    else:
        N = sys.argv[1]
        Pgd = sys.argv[2]
        Pcn = sys.argv[3]
        Pjc = sys.argv[4]
        Paa = sys.argv[5]
        Ppa = sys.argv[6]
        try:
            PERCENTAGE = sys.argv[7]
        except Exception:
            PERCENTAGE = 0.005

else:
    N = 200
    Pgd = 0.6
    Pcn = 0.6
    Pjc = 0.6
    Paa = 0.6
    Ppa = 0.6
    PERCENTAGE = 0.005
'''


# %%
N = 200
Pgd = 0.6
Pcn = 0.6
Pjc = 0.6
Paa = 0.6
Ppa = 0.6
PERCENTAGE = 0.005


# %%
ALL_RECORDS = 63497049
# Keep only 0.5% of samples.

skip = ceil(ALL_RECORDS*(1-PERCENTAGE))
print("Loading data...")
data = pd.read_csv('sx-stackoverflow.txt', sep=" ", header=0, names=['source', 'target', 'timestamp'], dtype={'source': np.int32, 'target': np.int32, 'timestamp': np.int32}, skiprows=skip)
print("Done!")


# %%
# 1 Calculating t_min t_max
t_min = data.iloc[0,2]
t_max = data.iloc[-1,2]
print(f"t_min: {t_min}, t_max: {t_max}")


# %%
# 2 Calculating time intervals

Dt = t_max - t_min
dt = ceil(Dt/N)

t = [t_min + j * dt for j in range(N + 1)]
#print(t)
T = [[t[i], t[i + 1] - 1] for i in range(N)]
T[-1][1] = t_max # eliminate remainders
print(f"N: {N}, Dt: {Dt}, dt: {dt}")
for i in range(len(T)):
    pass
    print(f"T[{i}, {i+1}]: [{T[i][0]}, {T[i][1]}]")


# %%
# 3 Calculating undirected Graphs
print(f"Calculating undirected graphs for {N} time intervals.")

G = []
ti = 0
g = nx.Graph()
for i, r in data.iterrows():
    if r['timestamp'] < T[ti][1]:
        g.add_edge(r['source'], r['target'])
    else:
        G.append(g)
        g = nx.Graph()
        g.add_edge(r['source'], r['target'])
        ti += 1
print("Done!")


# %%
'''
# Printing number of edges for each subgraph
for i in range(len(G)):
    print(len(G[i].edges))
'''


# %%
def get_centralities(func, G):

    if func == nx.in_degree_centrality or func == nx.out_degree_centrality:
        G = nx.DiGraph(G)

    if func == nx.eigenvector_centrality or func == nx.katz_centrality:
        centr = func(G, max_iter=10000)
    else:
        centr = func(G)

    return centr


# %%
# 4 Calculating centrality metrics.
degree_centralities = []
in_degree_centralities = []
out_degree_centralities = []
closeness_centralities = []
betweenness_centralities = []
eigenvector_centralities = []
katz_centralities = []

start = time.time()
print("Calculating degree centralities for each graph.")
for i in range(len(G)):
    degree_centralities.append(get_centralities(nx.degree_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')

start = time.time()
print("Calculating in degree centralities for each graph. (Converted into directed graphs)")
for i in range(len(G)):
    in_degree_centralities.append(get_centralities(nx.in_degree_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')

start = time.time()
print("Calculating out degree centralities for each graph. (Converted into directed graphs)")
for i in range(len(G)):
    out_degree_centralities.append(get_centralities(nx.out_degree_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')

start = time.time()
print("Calculating closeness centralities for each graph. (Estimated execution time: 14'')")
for i in range(len(G)):
    closeness_centralities.append(get_centralities(nx.closeness_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')
    
start = time.time()
print("Calculating betweenness centralities for each graph. (Estimated execution time: 250'')")
for i in range(len(G)):
    betweenness_centralities.append(get_centralities(nx.betweenness_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')

start = time.time()
print("Calculating eigenvector centralities for each graph. (Estimated execution time: 84'')")
for i in range(len(G)):
    eigenvector_centralities.append(get_centralities(nx.eigenvector_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')

start = time.time()
print("Calculating katz centralities for each graph. (Estimated execution time: 12'')")
for i in range(len(G)):
    katz_centralities.append(get_centralities(nx.katz_centrality, G[i]))
print("Done! Execution time:", time.time() - start, '\n')
        


# %%
def set_size(w,h, ax=None):
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def nine_plot_centralities(centralities, bins=None):
    fig, ax = plt.subplots(3, 3)
    set_size(10,10)
    
    for i in range(9):
        ax[i // 3, i % 3].hist(centralities[i].values(), bins='doane', align='left')
        


# %%
print("Plotting last nine degree centralities histograms.")
nine_plot_centralities(degree_centralities, 'doane')
print()


# %%
print("Plotting last nine in degree centralities histograms.")
nine_plot_centralities(in_degree_centralities)
print()


# %%
print("Plotting last nine out degree centralities histograms.")
nine_plot_centralities(out_degree_centralities)
print()


# %%
print("Plotting last nine closeness centralities histograms.")
nine_plot_centralities(closeness_centralities)
print()


# %%
print("Plotting last nine betweenness centralities histograms.")
nine_plot_centralities(betweenness_centralities)
print()


# %%
print("Plotting last nine eigenvector centralities histograms.")
nine_plot_centralities(eigenvector_centralities)
print()


# %%
print("Plotting last nine katz centralities histograms.")
nine_plot_centralities(katz_centralities)
print()


# %%
# 5 Calculating V*
print("Calculating V* for every two consecutive intervals.")
V_star = []
for i in range(1, N - 1):
    v_0 = set(G[i-1].nodes)
    v_1 = set(G[i].nodes)
    intersection = list(v_0 & v_1)
    V_star.append(intersection)
print("Done!\n")


# %%
# Calculating E*
# E[j][0] the previous time interval (set 7 in SocialNetworksAnalysisAssignment.pdf), 
# E[j][1] the next time interval (set 8 in SocialNetworksAnalysisAssignment.pdf)

print("Calculating E* for each time interval.")
E_star = []
j = 0
for i in range(1, N-1):
    e_0 = []
    for e in list(G[i-1].edges):
        u,v = e
        if u in V_star[j] and v in V_star[j]:
            e_0.append((u,v))
    e_1 = []
    for e in list(G[i].edges):
        u,v = e
        if u in V_star[j] and v in V_star[j]:
            e_1.append((u,v))
    E_star.append([e_0, e_1])
    j += 1
print("Done!\n")


# %%
def get_similarity_dict(func, G):
    if func == nx.all_pairs_shortest_path_length:
        sim_dict = dict(func(G))

    elif func == nx.common_neighbors:
        sim_dict = {}
        for u in G.nodes:
            dict_u = {}
            for v in G.nodes:
                cnbors  = func(G, u, v)
                n_cnbors = sum(1 for _ in cnbors)
                dict_u[v] = n_cnbors
            sim_dict[u] = dict_u

    elif func == nx.jaccard_coefficient or func == nx.preferential_attachment:
        results = func(G, ebunch=G.edges)
        sim_dict = {k: {v: s} for k,v,s in results}

    elif func == nx.adamic_adar_index:
        G = G.copy()
        G.remove_edges_from(nx.selfloop_edges(G))
        results = func(G, ebunch=G.edges)
        sim_dict = {k: {v: s} for k,v,s in results}

    else:
        raise Exception("Not implemented for this function")
    return sim_dict


# %%
# 6 Creating G*s, where G*_j = (V*_j, E*_j[0])
print("Calclating G* for each V*. [G*_j = (V*_j, E*_j[0])]")
G_star = []
for i in range(len(V_star)):
    g = nx.Graph()
    for node in V_star[i]:
        g.add_node(node)
    for source, target in E_star[i][0]:
        g.add_edge(source, target)
    G_star.append(g)
print("Done!")


# %%
start = time.time()
print("Calculating Sdg for each G*.")
Sgd = [get_similarity_dict(nx.all_pairs_shortest_path_length, G_star[i]) for i in range(len(G_star))]
print("Done! Execution time: ", time.time()-start, "\n")

start = time.time()
print("Calculating Scn for each G*. (Estimated execution time: 120'')")
Scn = [get_similarity_dict(nx.common_neighbors, G_star[i]) for i in range(len(G_star))]
print("Done! Execution time: ", time.time()-start, "\n")

start = time.time()
print("Calculating Sjc for each G*.")
Sjc = [get_similarity_dict(nx.jaccard_coefficient, G_star[i]) for i in range(len(G_star))]
print("Done! Execution time: ", time.time()-start, "\n")

start = time.time()
print("Calculating Saa for each G*.")
Saa = [get_similarity_dict(nx.adamic_adar_index, G_star[i]) for i in range(len(G_star))]
print("Done! Execution time: ", time.time()-start, "\n")

start = time.time()
print("Calculating Spa for each G*.")
Spa = [get_similarity_dict(nx.preferential_attachment, G_star[i]) for i in range(len(G_star))]
print("Done! Execution time: ", time.time()-start, "\n")


# %%
def get_min_max_value_of_similarity(similarity):
    val = []
    for u in similarity:
        for v in similarity[u]:
            val.append(similarity[u][v])
    return (min(val), max(val))


def get_decision(p, s):
    s_decision = []
    for i in range(len(s)):
        min_val, max_val = get_min_max_value_of_similarity(s[i])
        decision = []
        for u in s[i].keys():
            for v in s[i][u].keys():
                if u == v:
                    continue
                value = s[i][u][v]
                if value >= p * max_val:
                    decision.append((u,v))
        s_decision.append(decision)
    return s_decision


def predict(decision):
    right_prediction = []
    for i in range(len(decision)):
        edges_exist = 0
        for u,v in decision[i]:
            if (u,v) in E_star[i][1]:
                edges_exist +=1

        if edges_exist == 0 and len(decision[i]) == 0 :
            right_prediction.append(0.0)
        else:
            right_prediction.append(edges_exist/len(decision[i]))
    return right_prediction


def print_scores(prediction):
    for i in range(len(prediction)):
        print('V[%d,%d]: %.2f%%' % (i, i+2,prediction[i]))


# %%
Sgd_desicion = get_decision(Pgd, Sgd)
Sgd_predictions = predict(Sgd_desicion)
print(10*"=" + " Sgd right predictions " + 10*"=")
print_scores(Sgd_predictions)


# %%
Scn_desicion = get_decision(Pcn, Scn)
Scn_predictions = predict(Scn_desicion)
print(10*"=" + " Scn right predictions " + 10*"=")
print_scores(Scn_predictions)


# %%
Sjc_desicion = get_decision(Pjc, Sjc)
Sjc_predictions = predict(Sjc_desicion)
print(10*"=" + " Sjc right predictions " + 10*"=")
print_scores(Sjc_predictions)


# %%
Saa_desicion = get_decision(Paa, Saa)
Saa_predictions = predict(Saa_desicion)
print(10*"=" + " Saa right predictions " + 10*"=")
print_scores(Saa_predictions)


# %%
Spa_desicion = get_decision(Ppa, Spa)
Spa_predictions = predict(Spa_desicion)
print(10*"=" + " Spa right predictions " + 10*"=")
print_scores(Spa_predictions)


# %%



# %%



# %%


