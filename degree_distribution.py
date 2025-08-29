# Charts: matplotlib only, one chart per figure, no explicit colors/styles.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.decomposition import PCA

rng = np.random.default_rng(123)

# 1) Build a synthetic product–product graph
n_categories = 3  # keep small to allow distinct node shapes without colors
n_per_cat = 60
n = n_categories * n_per_cat
categories = np.repeat(np.arange(n_categories), n_per_cat)

# Node ids and labels
nodes = [f"P{i:04d}" for i in range(n)]
cat_map = {nodes[i]: int(categories[i]) for i in range(n)}

# Edge probabilities
p_intra = 0.10   # higher within category
p_inter = 0.01   # lower across categories

G = nx.Graph()
for i in range(n):
    G.add_node(nodes[i], category=int(categories[i]))

# Add edges stochastically
for i in range(n):
    for j in range(i+1, n):
        same = categories[i] == categories[j]
        p = p_intra if same else p_inter
        if rng.random() < p:
            # weight ~ co-purchase count
            w = 1 + rng.integers(1, 5)
            G.add_edge(nodes[i], nodes[j], weight=int(w))

# 2) Community detection (greedy modularity)
communities = list(greedy_modularity_communities(G))
comm_id = {}
for k, comm in enumerate(communities):
    for v in comm:
        comm_id[v] = k
nx.set_node_attributes(G, comm_id, "community")

# 3) Degree distribution figure
deg = np.array([d for _, d in G.degree()])
plt.figure()
plt.hist(deg, bins=20)
plt.xlabel("Node degree")
plt.ylabel("Count of products")
plt.title("Degree distribution (product–product graph)")
plt.tight_layout()
plt.savefig('degree_destribution.png')
plt.show()
