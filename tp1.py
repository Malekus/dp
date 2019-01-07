from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(threshold=np.nan)

x, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

print(x, np.array([k if k == 1 else -1 for k in y]))

plt.figure()
plt.scatter(np.array(x + np.column_stack((y,y)))[:,0], np.array(x - np.column_stack((y,y)))[:,1], c=np.array([k if k == 1 else -1 for k in y]))
plt.show()

def err_adal(d, w, x):
    return d - w * x

def adapt_adal(et, x):
    return -2 * et * x


def adaline(x, y, pas):
    W = np.array([[random.uniform(0.0, 1.0)] for _ in y])
    
    ecartTab = err_adalTab(y, W[:, -1], x)
    adaptTab = adapt_adal_Tab(y, np.array(ecartTab)[-1], x)
    
    W.append(W[:, -1] - pas * adapt_adal_Tab(y, W, x))
    
    return W



print(adaline(x, y, 0.1))


def err_adalTab(y, W, x):
    return np.array([toto[:, -1] - W[:,-1] * x[:,0], toto[:, -1] - W[:,-1] * x[:,0]])

def adapt_adal_Tab(y, et, x):
    return np.array([-2 * et[0] * x[:,0], -2 * et[1] * x[:,1]])
