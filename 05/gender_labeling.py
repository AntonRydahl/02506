# Example of max flow min cut for gender labeling problem

import maxflow
import numpy as np

#%%
d = np.array([179, 174, 182, 162, 175, 165]) # heights (data)
mu = [181, 165] # means of two classes
beta = 100 # weight of the prior term
w_s = (d-mu[0])**2 # source weight
w_t = (d-mu[1])**2 # sink weights
N = len(d) # number of graph nodes

# Create a graph with integer capacities.
g = maxflow.Graph[int]()
# Add (non-terminal) nodes and retrieve an index for each node
nodes = g.add_nodes(N)
# Create edges between nodes
for i in range(N-1):
    g.add_edge(nodes[i], nodes[i+1], beta, beta)
# Set the capacities of the terminal edges.
for i in range(N):
    g.add_tedge(nodes[i], (d[i]-mu[1])**2, (d[i]-mu[0])**2)
# Run the max flow algorithm
flow = g.maxflow()
print(f'Maximum flow: {flow}')

# displaying the results
labeling = [g.get_segment(nodes[i]) for i in range(N)]
gend = 'MF'

for i in range(0,N):
    print(f'Person {i} is estimated as {gend[labeling[i]]}') 







