import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_moons(n_samples=800, noise=0.05, random_state=1000)

    # Perform a Kernel PCA
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10.0, random_state=1000)
    X_pca = kpca.fit_transform(X)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 8))

    ax[0].scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax[0].scatter(X[Y == 1, 0], X[Y == 1, 1])
    ax[0].set_xlabel(r'$x_1$', fontsize=16)
    ax[0].set_ylabel(r'$x_2$', fontsize=16)
    ax[0].set_title('Original dataset', fontsize=16)

    ax[1].scatter(X_pca[Y == 0, 0], X_pca[Y == 0, 1])
    ax[1].scatter(X_pca[Y == 1, 0], X_pca[Y == 1, 1])
    ax[1].set_xlabel('First component', fontsize=16)
    ax[1].set_ylabel('Second component', fontsize=16)
    ax[1].set_title('RBF Kernel PCA projected dataset', fontsize=16)

    plt.show()