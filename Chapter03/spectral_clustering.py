import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import SpectralClustering, KMeans


# For reproducibility
np.random.seed(1000)


nb_samples = 2000


if __name__ == '__main__':
    # Create the dataset
    X0 = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, nb_samples), axis=1)
    Y0 = -2.0 - np.cos(2.0 * X0) + np.random.uniform(0.0, 2.0, size=(nb_samples, 1))

    X1 = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, nb_samples), axis=1)
    Y1 = 2.0 - np.cos(2.0 * X0) + np.random.uniform(0.0, 2.0, size=(nb_samples, 1))

    data_0 = np.concatenate([X0, Y0], axis=1)
    data_1 = np.concatenate([X1, Y1], axis=1)
    data = np.concatenate([data_0, data_1], axis=0)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)

    plt.show()

    # Perform the clustering
    km = KMeans(n_clusters=2, random_state=1000)
    sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=2.0, random_state=1000)

    Y_pred_km = km.fit_predict(data)
    Y_pred_sc = sc.fit_predict(data)

    # Show the results
    fig, ax = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    ax[0].scatter(data[:, 0], data[:, 1], c='b', s=5)

    ax[1].scatter(data[Y_pred_sc == 0, 0], data[Y_pred_sc == 0, 1], marker='o', s=5, c='b', label='Cluster 1')
    ax[1].scatter(data[Y_pred_sc == 1, 0], data[Y_pred_sc == 1, 1], marker='d', s=5, c='gray', label='Cluster 2')

    ax[2].scatter(data[Y_pred_km == 0, 0], data[Y_pred_km == 0, 1], marker='o', c='b', s=5, label='Cluster 1')
    ax[2].scatter(data[Y_pred_km == 1, 0], data[Y_pred_km == 1, 1], marker='d', c='gray', s=5, label='Cluster 2')

    ax[0].set_title('Dataset', fontsize=14)
    ax[0].set_xlabel(r'$x_0$', fontsize=14)
    ax[0].set_ylabel(r'$x_1$', fontsize=14)

    ax[1].set_title('Spectral Clustering', fontsize=14)
    ax[1].set_xlabel(r'$x_0$', fontsize=14)
    ax[1].legend()

    ax[2].set_title('K-Means', fontsize=14)
    ax[2].set_xlabel(r'$x_0$', fontsize=14)
    ax[2].legend()

    plt.show()