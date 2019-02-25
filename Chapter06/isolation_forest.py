import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE


# For reproducibility
np.random.seed(1000)


nb_samples = 2000
nb_test_samples = 200


if __name__ == '__main__':
    # Load the dataset
    wine = load_wine()
    X = wine['data'].astype(np.float64)

    # Normalize the dataset
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Train the isolation forest
    isf = IsolationForest(n_estimators=150, behaviour='new', contamination=0.01, random_state=1000)
    Y_pred = isf.fit_predict(X)

    print('Outliers in the training set: {}'.format(np.sum(Y_pred == -1)))

    # Create the test set
    X_test_1 = np.mean(X) + np.random.normal(0.0, 1.0, size=(50, 13))
    X_test_2 = np.mean(X) + np.random.normal(0.0, 2.0, size=(50, 13))
    X_test = np.concatenate([X_test_1, X_test_2], axis=0)

    Y_test = isf.predict(X_test) * 2

    Xf = np.concatenate([X, X_test], axis=0)
    Yf = np.concatenate([Y_pred, Y_test], axis=0)

    print(Yf[::-1])

    # Perform the t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=5, n_iter=5000, random_state=1000)
    X_tsne = tsne.fit_transform(Xf)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.scatter(X_tsne[Yf == 1, 0], X_tsne[Yf == 1, 1], marker='o', s=100, label='Inliers')
    ax.scatter(X_tsne[Yf == -1, 0], X_tsne[Yf == -1, 1], marker='x', s=100, label='Ouliers')
    ax.scatter(X_tsne[Yf == 2, 0], X_tsne[Yf == 2, 1], marker='^', s=80, label='Test inliers')
    ax.scatter(X_tsne[Yf == -2, 0], X_tsne[Yf == -2, 1], marker='v', s=80, label='Test ouliers')

    ax.set_xlabel(r'$x_1$', fontsize=14)
    ax.set_ylabel(r'$x_2$', fontsize=14)

    ax.legend(fontsize=14)

    plt.show()


