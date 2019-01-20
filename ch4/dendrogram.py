import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet


# For reproducibility
np.random.seed(1000)


nb_samples = 12
nb_centers = 4


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

    # Compute the distance matrix
    dm = pdist(X, metric='euclidean')

    # Show the dendrogram with Ward's linkage
    Z = linkage(dm, method='ward')

    fig, ax = plt.subplots(figsize=(12, 8))

    d = dendrogram(Z, show_leaf_counts=True, leaf_font_size=14, ax=ax)

    ax.set_xlabel('Samples', fontsize=14)
    ax.set_yticks(np.arange(0, 6.0, 0.25))

    plt.show()

    # Show the dendrogram with single linkage
    Z = linkage(dm, method='single')

    fig, ax = plt.subplots(figsize=(12, 8))

    d = dendrogram(Z, show_leaf_counts=True, leaf_font_size=14, ax=ax)

    ax.set_xlabel('Samples', fontsize=14)
    ax.set_yticks(np.arange(0, 2.0, 0.25))

    plt.show()

    # Print the cophenetic correlations
    cpc, cp = cophenet(linkage(dm, method='ward'), dm)
    print('CPC Ward\'s linkage: {:.3f}'.format(cpc))

    cpc, cp = cophenet(linkage(dm, method='single'), dm)
    print('CPC Single linkage: {:.3f}'.format(cpc))

    cpc, cp = cophenet(linkage(dm, method='complete'), dm)
    print('CPC Complete linkage: {:.3f}'.format(cpc))

    cpc, cp = cophenet(linkage(dm, method='average'), dm)
    print('CPC Average linkage: {:.3f}'.format(cpc))

