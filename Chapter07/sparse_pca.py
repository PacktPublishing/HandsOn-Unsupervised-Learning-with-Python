import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.decomposition import SparsePCA


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X = digits['data'] / np.max(digits['data'])

    # Perform a sparse PCA
    spca = SparsePCA(n_components=30, alpha=2.0, normalize_components=True, random_state=1000)
    spca.fit(X)

    # Show the components
    sns.set()

    fig, ax = plt.subplots(3, 10, figsize=(22, 8))

    for i in range(3):
        for j in range(10):
            ax[i, j].imshow(spca.components_[(3 * j) + i].reshape((8, 8)), cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()

    # Transform X[0]
    y = spca.transform(X[0].reshape(1, -1)).squeeze()

    # Show the absolute magnitudes
    fig, ax = plt.subplots(figsize=(22, 10))

    ax.bar(np.arange(1, 31, 1), np.abs(y))
    ax.set_xticks(np.arange(1, 31, 1))
    ax.set_xlabel('Component', fontsize=16)
    ax.set_ylabel('Coefficient (absolute values)', fontsize=16)

    plt.show()


