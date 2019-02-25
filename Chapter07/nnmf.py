import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.decomposition import NMF


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X = digits['data'] / np.max(digits['data'])

    # Perform a Non-negative matrix factorization
    nmf = NMF(n_components=50, alpha=2.0, l1_ratio=0.1, random_state=1000)
    nmf.fit(X)

    # Show the components
    sns.set()

    fig, ax = plt.subplots(5, 10, figsize=(22, 15))

    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(nmf.components_[(5 * j) + i].reshape((8, 8)), cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()

    # Transform X[0]
    y = nmf.transform(X[0].reshape(1, -1)).squeeze()

    # Show the absolute magnitudes
    fig, ax = plt.subplots(figsize=(22, 10))

    ax.bar(np.arange(1, 51, 1), np.abs(y))
    ax.set_xticks(np.arange(1, 51, 1))
    ax.set_xlabel('Component', fontsize=16)
    ax.set_ylabel('Coefficient (absolute values)', fontsize=16)

    plt.show()


