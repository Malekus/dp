from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.nan)

x, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
x = np.column_stack((x, np.ones(x.shape[0])))
print(x, np.array([k if k == 1 else -1 for k in y]))

plt.figure()
plt.scatter(np.array(x + np.column_stack((y,y)))[:,0], np.array(x - np.column_stack((y,y)))[:,1], c=np.array([k if k == 1 else -1 for k in y]))
plt.show()

W = [random.uniform(0.0, 1.0) for _ in range(x.shape[1])]

def hasardW(x):
    return np.array([random.uniform(0.0, 1.0) for _ in range(x.shape[1])])

def err_dal(w, x, y):
    return y - np.dot(x, w)



def grad_adal(w, x, et):
    return -2 * np.dot(et, x)

def adapt_adal(w, ra, pas):
    return w - pas * ra


def adaline(x, y, tauxErreur=0.5, pas=0.1):
    W = hasardW(x)
    WW = []
    while(np.any(np.absolute(err_dal(W, x, y)) < 0.5)):
        WW.append(W)
        et = err_dal(W, x, y)
        ra = grad_adal(W, x, et)
        W = adapt_adal(W, ra, pas)
    return np.array(WW)

adaline = []
adaline = adaline(x, y, pas=0.1)

len(adaline)

pca = PCA(n_components=1).fit_transform(adaline[0])

plt.figure()
plt.plot(range(adaline[1]), pca[:,0])
plt.show()

W = hasardW(x)
True == np.all(np.power(W, 2) < 0.5)

np.abs(err_dal(W, x, y))