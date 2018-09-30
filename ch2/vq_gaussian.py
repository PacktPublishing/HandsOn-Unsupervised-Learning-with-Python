import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist


# For reproducibility
np.random.seed(1000)


nb_samples = 1000
n_vectors = 16
delta = 0.05
n_iterations = 1000


if __name__ == '__main__':
    # Initialize the dataset and the vectors
    data = np.random.normal(0.0, 1.5, size=(nb_samples, 2))
    qv = np.random.normal(0.0, 1.5, size=(n_vectors, 2))

    # Show the initial configuration
    sns.set()

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(data[:, 0], data[:, 1], marker='d', s=15, label='Samples')
    ax.scatter(qv[:, 0], qv[:, 1], s=100, label='QVs')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()

    plt.show()

    # Perform the computation
    for i in range(n_iterations):
        for p in data:
            distances = cdist(qv, np.expand_dims(p, axis=0))
            qvi = np.argmin(distances)
            alpha = p - qv[qvi]
            qv[qvi] += (delta * alpha)

    # Show the final configuration
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(data[:, 0], data[:, 1], marker='d', s=20, label='Samples')
    ax.scatter(qv[:, 0], qv[:, 1], s=100, label='QVs')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()

    plt.show()