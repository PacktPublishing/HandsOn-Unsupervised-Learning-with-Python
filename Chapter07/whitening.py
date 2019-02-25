import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs


# For reproducibility
np.random.seed(1000)


def zero_center(X):
    return X - np.mean(X, axis=0)


def whiten(X, correct=True):
    Xc = zero_center(X)
    _, L, V = np.linalg.svd(Xc)
    W = np.dot(V.T, np.diag(1.0 / L))
    return np.dot(Xc, W) * np.sqrt(X.shape[0]) if correct else 1.0


if __name__ == '__main__':
    # Create the dataset
    X, _ = make_blobs(n_samples=300, centers=1, cluster_std=2.5, random_state=1000)

    print(np.cov(X.T))

    Xw = whiten(X)

    # Show the plots
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 8))

    ax[0].scatter(X[:, 0], X[:, 1])
    ax[0].set_xlim([-10, 10])
    ax[0].set_ylim([-16, 16])
    ax[0].set_xlabel(r'$x_1$', fontsize=16)
    ax[0].set_ylabel(r'$x_2$', fontsize=16)
    ax[0].set_title('Original dataset', fontsize=16)

    ax[1].scatter(Xw[:, 0], Xw[:, 1])
    ax[1].set_xlim([-10, 10])
    ax[1].set_ylim([-16, 16])
    ax[1].set_xlabel(r'$x_1$', fontsize=16)
    ax[1].set_ylabel(r'$x_2$', fontsize=16)
    ax[1].set_title('Whitened dataset', fontsize=16)

    plt.show()

