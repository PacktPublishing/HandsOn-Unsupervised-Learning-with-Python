import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


# For reproducibility
np.random.seed(1000)


nb_samples = 1000
nb_clusters = 8

metric = 'minkowski'
p = 7
tolerance = 0.001


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=nb_clusters,
                      cluster_std=1.2, center_box=[-5.0, 5.0], random_state=1000)

    # Perform K-Means clustering
    km = KMeans(n_clusters=nb_clusters, random_state=1000)
    C_km = km.fit_predict(X)

    print('Adjusted Rand score K-Means: {}'.format(adjusted_rand_score(Y, C_km)))

    # Perform K-Medoids clustering
    C = np.random.randint(0, nb_clusters, size=(X.shape[0],), dtype=np.int32)
    mu_idxs = np.zeros(shape=(nb_clusters, X.shape[1]))

    mu_copy = np.ones_like(mu_idxs)

    while np.linalg.norm(mu_idxs - mu_copy) > tolerance:
        for i in range(nb_clusters):
            Di = squareform(pdist(X[C == i], metric=metric, p=p))
            SDi = np.sum(Di, axis=1)

            mu_copy[i] = mu_idxs[i].copy()
            idx = np.argmin(SDi)
            mu_idxs[i] = X[C == i][idx].copy()

        C = np.argmin(cdist(X, mu_idxs, metric=metric, p=p), axis=1)

    print('Adjusted Rand score K-Medoids: {}'.format(adjusted_rand_score(Y, C)))

    # Show the final results
    sns.set()

    fig, ax = plt.subplots(1, 3, figsize=(26, 8), sharey=True)

    for i in range(nb_clusters):
        ax[0].scatter(X[Y == i, 0], X[Y == i, 1], label='Blob {}'.format(i + 1))
        ax[1].scatter(X[C_km == i, 0], X[C_km == i, 1], label='Cluster {}'.format(i + 1))
        ax[2].scatter(X[C == i, 0], X[C == i, 1], label='Cluster {}'.format(i + 1))

    ax[0].set_xlabel(r'$x_0$', fontsize=22)
    ax[1].set_xlabel(r'$x_0$', fontsize=22)
    ax[2].set_xlabel(r'$x_0$', fontsize=22)
    ax[0].set_ylabel(r'$x_1$', fontsize=22)

    ax[0].set_title('Ground truth', fontsize=18)
    ax[1].set_title('K-Means', fontsize=18)
    ax[2].set_title('K-Medoids with Minkowski metric and p=7', fontsize=18)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.show()

