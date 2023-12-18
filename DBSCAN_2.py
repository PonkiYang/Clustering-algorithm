import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
np.random.seed(2021)

data = np.ones([1005,2])
data[:1000] = make_moons(n_samples=1000,noise=0.05,random_state=2022)[0]
data[1000:] = [[-1,-0.5],
                [-0.5,-1],
                [-1,1.5],
                [2.5,-0.5],
                [2,1.5]]
print(data.shape)
plt.scatter(data[:,0],data[:,1],color="c")
plt.show()