import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.decomposition import FastICA


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X = digits['data'] / np.max(digits['data'])

    # Perform the fast ICA
    ica = FastICA(n_components=50, max_iter=10000, tol=1e-5, random_state=1000)
    ica.fit(X)

    # Show the components
    sns.set()

    fig, ax = plt.subplots(5, 10, figsize=(15, 10))

    for i in range(5):
        for j in range(10):
            ax[i, j].imshow(ica.components_[(2 * j) + i].reshape((8, 8)), cmap='gray')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.show()

    # Load the Olivetti faces dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=1000)

    # Show the first 10 faces
    fig, ax = plt.subplots(1, 10, figsize=(22, 12))

    for i in range(10):
        ax[i].imshow(faces['images'][i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()

    # Perform the fast ICA
    for n in (100, 350):
        ica = FastICA(n_components=n, max_iter=10000, tol=1e-5, random_state=1000)
        ica.fit(faces['data'])

        # Show the first 50 components
        fig, ax = plt.subplots(5, 10, figsize=(15, 10))

        for i in range(5):
            for j in range(10):
                ax[i, j].imshow(ica.components_[(5 * j) + i].reshape((64, 64)), cmap='gray')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

        plt.show()

