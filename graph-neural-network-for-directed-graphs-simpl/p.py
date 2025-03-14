import torch
import numpy
from datasets.graph_data_reader import GraphData

A = torch._as_tensor_fullprec(
   [
       [0, 1, 0],
       [0, 0, 1],
       [1, 0, 0]
   ] 
)
print(torch.sum(A, dim=1))
print(numpy.diag([1,2,3]))
print(GraphData.ad2MagL(A, 0.25, True))