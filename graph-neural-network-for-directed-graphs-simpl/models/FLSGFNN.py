r"""
    FAST LOCALIZED SPECTRAL GRAPH NEURAL NETWORK

In this model we use a fast localized spectral graph filtering
in the sense of: 

We extend this method to directed graphs using the magnetic 
laplacian. In the paper:
it's defined for a neural network on node-level and edge-level. 
Thus we extend it to graph level with  a read-out function, 
averege, maximum or max. Future improvements could be adding
some pooling layers so we have a more meaninful read-out. Now 
we use simple readouts because we want to study the performance
of spectral filterings on a graph-level task for directed graphs
"""

