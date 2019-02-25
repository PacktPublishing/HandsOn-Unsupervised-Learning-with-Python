import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the distance matrix
    distances = np.zeros(shape=(8, 100))

    for i in range(1, distances.shape[0] + 1):
        for j in range(1, distances.shape[1] + 1):
            distances[i - 1, j - 1] = np.log(cdist(np.zeros(shape=(1, j)), np.ones(shape=(1, j)),
                                                   metric='minkowski', p=i)[0][0])

    # Show the distances
    sns.set()

    fig, ax = plt.subplots(figsize=(16, 9))

    for i in range(distances.shape[0]):
        ax.plot(np.arange(1, distances.shape[1] + 1, 1), distances[i], label='p={}'.format(i))

    ax.set_xlabel('Dimensionality', fontsize=14)
    ax.set_ylabel('Minkowski distances (log-scale)', fontsize=14)
    ax.legend()
    ax.set_xticks(np.arange(1, distances.shape[1] + 2, 5))
    ax.set_yticks(np.arange(0, 5, 0.5))

    plt.show()



