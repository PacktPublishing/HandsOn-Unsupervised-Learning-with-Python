import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


# For reproducibility
np.random.seed(1000)


nb_clusters = 8
nb_samples = 2000
batch_size = 50


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=nb_clusters,
                      cluster_std=0.25, center_box=[-1.5, 1.5], shuffle=True, random_state=100)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(nb_clusters):
        ax.scatter(X[Y == i, 0], X[Y == i, 1], label='Blob {}'.format(i + 1))

    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)
    ax.legend()

    plt.show()

    # Perform a K-Means clustering
    km = KMeans(n_clusters=nb_clusters, random_state=1000)
    Y_pred_km = km.fit_predict(X)

    print('Adjusted Rand score: {}'.format(adjusted_rand_score(Y, Y_pred_km)))

    # Perform the online clustering
    mbkm = MiniBatchKMeans(n_clusters=nb_clusters, batch_size=batch_size, reassignment_ratio=0.001, random_state=1000)
    birch = Birch(n_clusters=nb_clusters, threshold=0.2, branching_factor=350)

    scores_mbkm = []
    scores_birch = []

    for i in range(0, nb_samples, batch_size):
        X_batch, Y_batch = X[i:i + batch_size], Y[i:i + batch_size]

        mbkm.partial_fit(X_batch)
        birch.partial_fit(X_batch)

        scores_mbkm.append(adjusted_rand_score(Y[:i + batch_size], mbkm.predict(X[:i + batch_size])))
        scores_birch.append(adjusted_rand_score(Y[:i + batch_size], birch.predict(X[:i + batch_size])))

    Y_pred_mbkm = mbkm.predict(X)
    Y_pred_birch = birch.predict(X)

    print('Adjusted Rand score Mini-Batch K-Means: {}'.format(adjusted_rand_score(Y, Y_pred_mbkm)))
    print('Adjusted Rand score BIRCH: {}'.format(adjusted_rand_score(Y, Y_pred_birch)))

    # Show the incremental Adjusted Rand scores
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(range(0, nb_samples, batch_size), scores_mbkm, label='Mini-Batch K-Means')
    ax.plot(range(0, nb_samples, batch_size), scores_birch, label='Birch')

    ax.set_xlabel('Number of samples', fontsize=14)
    ax.set_ylabel('Incremental Adjusted Rand score', fontsize=14)
    ax.legend()

    plt.show()

