import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score

from skfuzzy.cluster import cmeans


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X = digits['data'] / 255.0
    Y = digits['target']

    # Perform a preliminary analysis
    Ws = []
    pcs = []

    for m in np.linspace(1.05, 1.5, 5):
        fc, W, _, _, _, _, pc = cmeans(X.T, c=10, m=m, error=1e-6, maxiter=20000, seed=1000)
        Ws.append(W)
        pcs.append(pc)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    for i, m in enumerate(np.linspace(1.05, 1.5, 5)):
        ax[i].bar(np.arange(10), -np.log(Ws[i][:, 0]))
        ax[i].set_xticks(np.arange(10))
        ax[i].set_title(r'$m={}, P_C={:.2f}$'.format(m, pcs[i]))

    ax[0].set_ylabel(r'$-log(w_0j)$')

    plt.show()

    # Perform the clustering
    fc, W, _, _, _, _, pc = cmeans(X.T, c=10, m=1.2, error=1e-6, maxiter=20000, seed=1000)
    Mu = fc.reshape((10, 8, 8))

    # Show the centroids
    fig, ax = plt.subplots(1, 10, figsize=(20, 4))

    for i in range(10):
        ax[i].imshow(Mu[i] * 255, cmap='gray')
        ax[i].grid(False)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

    # Show the assignments for X[0]
    print(W[:, 0])

    # Compute the adjusted Rand score
    Y_pred = np.argmax(W.T, axis=1)

    print(adjusted_rand_score(Y, Y_pred))

    im = np.argmin(np.std(W.T, axis=1))

    print(im)
    print(Y[im])
    print(W[:, im])