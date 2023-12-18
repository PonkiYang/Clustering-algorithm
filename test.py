import numpy as np


train_X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
per = np.random.permutation(train_X.shape[0])
new_train_X = train_X[per, :]
print(new_train_X)