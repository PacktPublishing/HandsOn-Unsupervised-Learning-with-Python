import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram


# For reproducibility
np.random.seed(1000)


nb_samples = 50
nb_centers = 8


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, center_box=[-1, 1], centers=nb_centers, random_state=1000)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, x in enumerate(X):
        ax.scatter(x[0], x[1], s=120)
        ax.annotate('%d' % i, xy=(x[0] + 0.05, x[1] + 0.05), fontsize=12)

    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)

    plt.show()

    # Show the dendrogram with average linkage
    dm = pdist(X, metric='euclidean')
    Z = linkage(dm, method='average')

    fig, ax = plt.subplots(figsize=(20, 10))

    d = dendrogram(Z, orientation='right', truncate_mode='lastp', p=20, ax=ax)

    ax.set_xlabel('Dissimilarity', fontsize=18)
    ax.set_ylabel('Samples', fontsize=18)

    plt.show()

    # Perform the standard clustering
    ag = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='average')
    Y_pred = ag.fit_predict(X)

    # Show the results
    fig, ax = plt.subplots(figsize=(12, 8))

    clusters = set()

    for i, x in enumerate(X):
        y = Y_pred[i]
        if y in clusters:
            label = False
        else:
            clusters.add(y)
            label = True

        ax.scatter(x[0], x[1], s=120, c=cm.Set1(y), label='Cluster {}'.format(y + 1) if label else None)
        ax.annotate('%d' % i, xy=(x[0] + 0.05, x[1] + 0.05), fontsize=12)

    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)
    ax.legend()

    plt.show()

    # Build the connectivity matrix
    cma = kneighbors_graph(X, n_neighbors=2)

    # Perform the clustering with connectivity constraints
    ag = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='average', connectivity=cma)
    Y_pred = ag.fit_predict(X)

    # Show the new results
    fig, ax = plt.subplots(figsize=(12, 8))

    clusters = set()

    for i, x in enumerate(X):
        y = Y_pred[i]
        if y in clusters:
            label = False
        else:
            clusters.add(y)
            label = True

        ax.scatter(x[0], x[1], s=120, c=cm.Set1(y), label='Cluster {}'.format(y + 1) if label else None)
        ax.annotate('%d' % i, xy=(x[0] + 0.05, x[1] + 0.05), fontsize=12)

    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)
    ax.legend()

    plt.show()






