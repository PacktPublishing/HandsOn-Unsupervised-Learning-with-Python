import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    X, Y = make_blobs(n_samples=2000, n_features=2, centers=12,
                      cluster_std=0.05, center_box=[-5, 5], random_state=100)

    # Show the blobs
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(12):
        ax.scatter(X[Y == i, 0], X[Y == i, 1], label='Blob {}'.format(i + 1))

    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()

    plt.show()

    # Compute the inertia
    inertias = []

    for i in range(2, 21):
        km = KMeans(n_clusters=i, max_iter=1000, random_state=1000)
        km.fit(X)
        inertias.append(km.inertia_)

    # Show the plot inertia vs. no. clusters
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(np.arange(2, 21, 1), inertias)

    ax.set_xlabel('Number of clusters', fontsize=14)
    ax.set_ylabel('Inertia', fontsize=14)
    ax.set_xticks(np.arange(2, 21, 1))

    plt.show()