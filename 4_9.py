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

# 4) Network plot with different node shapes per community
pos = nx.spring_layout(G, seed=7, k=0.25)  # fixed for reproducibility

# Define shapes for up to 3 communities
shapes = ["o", "s", "^"]
plt.figure()
for k in range(min(len(communities), len(shapes))):
    nodes_k = [v for v in G.nodes if G.nodes[v]["community"] == k]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_k, node_size=40, node_shape=shapes[k])
# Draw edges once
nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
plt.title("Co-purchase graph (node shapes = detected communities)")
plt.axis("off")
plt.tight_layout()
plt.savefig("network_plot.png")
plt.show()

# 5) Simple GCN-like message passing demo
# Build initial node features as one-hot category + small noise (d=5)
d = 5
X = np.zeros((n, d))
for i in range(n):
    X[i, categories[i] % d] = 1.0
X += 0.05 * rng.normal(size=X.shape)

# Adjacency with self-loops
A = nx.to_numpy_array(G)
A_hat = A + np.eye(n)
# Degree normalization D^{-1/2} A_hat D^{-1/2}
deg_hat = np.sum(A_hat, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg_hat, 1e-8)))
S = D_inv_sqrt @ A_hat @ D_inv_sqrt

# One message-passing layer with a random linear transform
W = rng.normal(scale=0.5, size=(d, 2))  # project to 2 dims for plotting after smoothing
H0 = X.copy()
H1 = np.maximum(0.0, S @ H0 @ W)  # ReLU

# PCA baseline to 2D for the raw features (for comparison)
pca = PCA(n_components=2, random_state=0)
Z0 = pca.fit_transform(H0)  # before smoothing
Z1 = H1  # after smoothing (already 2D)

# 6) Scatter plots before and after message passing
plt.figure()
plt.scatter(Z0[:,0], Z0[:,1], s=10, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Node features — before smoothing (PCA 2D)")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(Z1[:,0], Z1[:,1], s=10, alpha=0.7)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("Node features — after one message-passing layer")
plt.tight_layout()
plt.show()

# 5) Simple GCN-like message passing demo
# Build initial node features as one-hot category + small noise (d=5)
d = 5
X = np.zeros((n, d))
for i in range(n):
    X[i, categories[i] % d] = 1.0
X += 0.05 * rng.normal(size=X.shape)

# Adjacency with self-loops
A = nx.to_numpy_array(G)
A_hat = A + np.eye(n)
# Degree normalization D^{-1/2} A_hat D^{-1/2}
deg_hat = np.sum(A_hat, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg_hat, 1e-8)))
S = D_inv_sqrt @ A_hat @ D_inv_sqrt

# One message-passing layer with a random linear transform
W = rng.normal(scale=0.5, size=(d, 2))  # project to 2 dims for plotting after smoothing
H0 = X.copy()
H1 = np.maximum(0.0, S @ H0 @ W)  # ReLU

# PCA baseline to 2D for the raw features (for comparison)
pca = PCA(n_components=2, random_state=0)
Z0 = pca.fit_transform(H0)  # before smoothing
Z1 = H1  # after smoothing (already 2D)

# 6) Scatter plots before and after message passing
plt.figure()
plt.scatter(Z0[:,0], Z0[:,1], s=10, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Node features — before smoothing (PCA 2D)")
plt.tight_layout()
plt.savefig("message_before.png")
plt.show()

plt.figure()
plt.scatter(Z1[:,0], Z1[:,1], s=10, alpha=0.7)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("Node features — after one message-passing layer")
plt.tight_layout()
plt.savefig("message_after.png")
plt.show()

# 7) Small tables: top edges and community sizes
edges = []
for u, v, dct in G.edges(data=True):
    edges.append((u, v, dct.get("weight", 1)))
edges_df = pd.DataFrame(edges, columns=["item_u","item_v","weight"]).sort_values("weight", ascending=False)

comm_sizes = (
    pd.Series({k: len(c) for k, c in enumerate(communities)})
      .rename("n_nodes")
      .reset_index()
      .rename(columns={"index":"community"})
)

print("Top co-purchase edges (by weight, sample)", end="\n")
print(edges_df.head(),end="\n\n")
print("Detected community sizes", end="\n")
print(comm_sizes)
