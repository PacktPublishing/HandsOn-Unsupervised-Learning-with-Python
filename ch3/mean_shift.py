import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MeanShift


# For reproducibility
np.random.seed(1000)


nb_samples = 500
mss = []
Y_preds = []
bandwidths = [0.9, 1.0, 1.2, 1.5]


if __name__ == '__main__':
    # Create the dataset
    data_1 = np.random.multivariate_normal([-2.0, 0.0], np.diag([1.0, 0.5]), size=(nb_samples,))
    data_2 = np.random.multivariate_normal([0.0, 2.0], np.diag([1.5, 1.5]), size=(nb_samples,))
    data_3 = np.random.multivariate_normal([2.0, 0.0], np.diag([0.5, 1.0]), size=(nb_samples,))

    data = np.concatenate([data_1, data_2, data_3], axis=0)

    # Show the original dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlabel(r'$x_0$', fontsize=14)
    ax.set_ylabel(r'$x_1$', fontsize=14)

    plt.show()

    # Perform the clustering with different bandwidths
    for b in bandwidths:
        ms = MeanShift(bandwidth=b)
        Y_preds.append(ms.fit_predict(data))
        mss.append(ms)

    # Show the results
    fig, ax = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    for j, b in enumerate(bandwidths):
        for i in range(mss[j].cluster_centers_.shape[0]):
            ax[j].scatter(data[Y_preds[j] == i, 0], data[Y_preds[j] == i, 1], marker='o', s=15,
                          label='Cluster {}'.format(i + 1))

        ax[j].set_xlabel(r'$x_0$', fontsize=14)
        ax[j].set_title('Bandwidth: {}'.format(b), fontsize=14)
        ax[j].legend()

    ax[0].set_ylabel(r'$x_1$', fontsize=14)

    plt.show()