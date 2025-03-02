import numpy as np

def loss(z):
    if z >= 1: 
        return 0
    else: 
        return 1-z
    
def loss2(z): return z*z/2

dots = [[np.array([1,0,1]),2], [np.array([1,1,1]),2.7], [np.array([1,1,-1]),-0.7],[np.array([-1,1,1]),2]]


R = 0
X = np.array([0,1,2])
for D in dots:
    dot, y = D[0], D[1]
    R += loss2(y - np.dot(dot, X))

print(R/4)