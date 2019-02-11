import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, FactorAnalysis


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=1000)
    X = faces['data']
    Xz = X - np.mean(X, axis=0)

    # Create a noisy version
    C = np.diag(np.random.uniform(0.0, 0.1, size=Xz.shape[1]))
    Xnz = Xz + np.random.multivariate_normal(np.zeros(shape=Xz.shape[1]), C, size=Xz.shape[0])

    # Show some samples
    sns.set()

    fig, ax = plt.subplots(2, 10, figsize=(22, 6))

    Xn = Xnz + np.mean(X, axis=0)

    for i in range(10):
        ax[0, i].imshow(X[i].reshape((64, 64)), cmap='gray')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])

        ax[1, i].imshow(Xn[i].reshape((64, 64)), cmap='gray')
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])

    plt.show()

    # Perform the evaluations
    pca = PCA(n_components=128, random_state=1000)
    pca.fit(Xz)
    print('PCA log-likelihood(Xz): {}'.format(pca.score(Xz)))

    pcan = PCA(n_components=128, random_state=1000)
    pcan.fit(Xnz)
    print('PCA log-likelihood(Xnz): {}'.format(pcan.score(Xnz)))

    fa = FactorAnalysis(n_components=128, random_state=1000)
    fa.fit(Xnz)
    print('Factor Analysis log-likelihood(Xnz): {}'.format(fa.score(Xnz)))
