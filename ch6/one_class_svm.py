import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# For reproducibility
np.random.seed(1000)


nb_samples = 2000
nb_test_samples = 200


if __name__ == '__main__':
    # Generate the dataset
    X = np.empty(shape=(nb_samples + nb_test_samples, 2))

    X[:nb_samples] = np.random.multivariate_normal([15, 160], np.diag([1.5, 10]), size=nb_samples)
    X[nb_samples:, 0] = np.random.uniform(11, 19, size=nb_test_samples)
    X[nb_samples:, 1] = np.random.uniform(120, 210, size=nb_test_samples)

    # Normalize the dataset
    ss = StandardScaler()
    Xs = ss.fit_transform(X)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(13, 8))

    ax.scatter(Xs[nb_samples:, 0], Xs[nb_samples:, 1], marker='^', s=80, label='Test samples')
    ax.scatter(Xs[:nb_samples, 0], Xs[:nb_samples, 1], label='Inliers')

    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('Height', fontsize=14)

    ax.legend(fontsize=14)

    plt.show()

    # Train the One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.2)
    Ys = ocsvm.fit_predict(Xs)

    # Show the results
    fig, ax = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

    ax[0].scatter(Xs[Ys == -1, 0], Xs[Ys == -1, 1], marker='x', s=100, label='Ouliers')
    ax[0].scatter(Xs[Ys == 1, 0], Xs[Ys == 1, 1], marker='o', label='Inliers')

    ax[1].scatter(Xs[Ys == -1, 0], Xs[Ys == -1, 1], marker='x', s=100)

    ax[0].set_xlabel('Age', fontsize=16)
    ax[0].set_ylabel('Height', fontsize=16)

    ax[1].set_xlabel('Age', fontsize=16)

    ax[0].set_title('All samples', fontsize=16)
    ax[1].set_title('Outliers', fontsize=16)

    ax[0].legend(fontsize=16)

    plt.show()

