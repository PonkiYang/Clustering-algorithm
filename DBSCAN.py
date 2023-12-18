import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

np.random.seed(2021)

# 整理数据
data = np.ones([1005, 2])
data[:1000] = make_moons(n_samples=1000, noise=0.05, random_state=2022)[0]
data[1000:] = [[-1, -0.5],
               [-0.5, -1],
               [-1, 1.5],
               [2.5, -0.5],
               [2, 1.5]]
print(data.shape)
plt.scatter(data[:, 0], data[:, 1], color="c")
plt.show()


# 选择eps和min_samples
def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)


k = 3  # 此处k取 2*2 -1
k_dist = select_MinPts(data, k)
k_dist.sort()
plt.plot(np.arange(k_dist.shape[0]), k_dist[::-1])
plt.show()  # 展示原始数据

# 建立模型
dbscan_model = DBSCAN(eps=0.1, min_samples=k + 1)
label = dbscan_model.fit_predict(data)
class_1 = []
class_2 = []
noise = []
for index, value in enumerate(label):
    if value == 0:
        class_1.append(index)
    elif value == 1:
        class_2.append(index)
    elif value == -1:
        noise.append(index)
plt.scatter(data[class_1, 0], data[class_1, 1], color="g", label="class 1")
plt.scatter(data[class_2, 0], data[class_2, 1], color="b", label="class 2")
plt.scatter(data[noise, 0], data[noise, 1], color="r", label="noise")
plt.legend()
plt.show()
