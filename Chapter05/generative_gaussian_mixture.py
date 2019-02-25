import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs

from matplotlib.patches import Ellipse

from scipy.stats import multivariate_normal


# For reproducibility
np.random.seed(1000)


nb_samples = 500
nb_unlabeled = 400
nb_iterations = 10


m1 = np.array([-2.0, -2.5])
c1 = np.array([[1.0, 1.0],
               [1.0, 2.0]])
q1 = 0.5

m2 = np.array([1.0, 3.0])
c2 = np.array([[2.0, -1.0],
               [-1.0, 3.5]])
q2 = 0.5


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=2, cluster_std=1.5, random_state=100)

    unlabeled_idx = np.random.choice(np.arange(0, nb_samples, 1), replace=False, size=nb_unlabeled)
    Y[unlabeled_idx] = -1

    # Show the initial configuration
    w1, v1 = np.linalg.eigh(c1)
    w2, v2 = np.linalg.eigh(c2)

    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)

    a1 = np.arccos(np.dot(nv1[:, 1], [1.0, 0.0]) / np.linalg.norm(nv1[:, 1])) * 180.0 / np.pi
    a2 = np.arccos(np.dot(nv2[:, 1], [1.0, 0.0]) / np.linalg.norm(nv2[:, 1])) * 180.0 / np.pi

    sns.set()

    fig, ax = plt.subplots(figsize=(22, 12))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=80, marker='o', label='Class 1')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=80, marker='d', label='Class 2')
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], s=100, marker='x', label='Unlabeled')

    g1 = Ellipse(xy=m1, width=w1[1] * 3, height=w1[0] * 3, fill=False, linestyle='dashed', angle=a1, color='black',
                 linewidth=1)
    g1_1 = Ellipse(xy=m1, width=w1[1] * 2, height=w1[0] * 2, fill=False, linestyle='dashed', angle=a1, color='black',
                   linewidth=2)
    g1_2 = Ellipse(xy=m1, width=w1[1] * 1.4, height=w1[0] * 1.4, fill=False, linestyle='dashed', angle=a1,
                   color='black', linewidth=3)

    g2 = Ellipse(xy=m2, width=w2[1] * 3, height=w2[0] * 3, fill=False, linestyle='dashed', angle=a2, color='black',
                 linewidth=1)
    g2_1 = Ellipse(xy=m2, width=w2[1] * 2, height=w2[0] * 2, fill=False, linestyle='dashed', angle=a2, color='black',
                   linewidth=2)
    g2_2 = Ellipse(xy=m2, width=w2[1] * 1.4, height=w2[0] * 1.4, fill=False, linestyle='dashed', angle=a2,
                   color='black', linewidth=3)

    ax.set_xlabel(r'$x_0$', fontsize=16)
    ax.set_ylabel(r'$x_1$', fontsize=16)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)

    ax.legend(fontsize=16)

    plt.show()

    # Train the model
    for i in range(nb_iterations):
        Pij = np.zeros((nb_samples, 2))

        for i in range(nb_samples):

            if Y[i] == -1:
                p1 = multivariate_normal.pdf(X[i], m1, c1, allow_singular=True) * q1
                p2 = multivariate_normal.pdf(X[i], m2, c2, allow_singular=True) * q2
                Pij[i] = [p1, p2] / (p1 + p2)
            else:
                Pij[i, :] = [1.0, 0.0] if Y[i] == 0 else [0.0, 1.0]

        n = np.sum(Pij, axis=0)
        m = np.sum(np.dot(Pij.T, X), axis=0)

        m1 = np.dot(Pij[:, 0], X) / n[0]
        m2 = np.dot(Pij[:, 1], X) / n[1]

        q1 = n[0] / float(nb_samples)
        q2 = n[1] / float(nb_samples)

        c1 = np.zeros((2, 2))
        c2 = np.zeros((2, 2))

        for t in range(nb_samples):
            c1 += Pij[t, 0] * np.outer(X[t] - m1, X[t] - m1)
            c2 += Pij[t, 1] * np.outer(X[t] - m2, X[t] - m2)

        c1 /= n[0]
        c2 /= n[1]

    print('Gaussian 1:')
    print(q1)
    print(m1)
    print(c1)

    print('\nGaussian 2:')
    print(q2)
    print(m2)
    print(c2)

    # Show the final configuration
    # Show the initial configuration
    w1, v1 = np.linalg.eigh(c1)
    w2, v2 = np.linalg.eigh(c2)

    nv1 = v1 / np.linalg.norm(v1)
    nv2 = v2 / np.linalg.norm(v2)

    a1 = np.arccos(np.dot(nv1[:, 1], [1.0, 0.0]) / np.linalg.norm(nv1[:, 1])) * 180.0 / np.pi
    a2 = np.arccos(np.dot(nv2[:, 1], [1.0, 0.0]) / np.linalg.norm(nv2[:, 1])) * 180.0 / np.pi

    sns.set()

    fig, ax = plt.subplots(figsize=(22, 12))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=80, marker='o', label='Class 1')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=80, marker='d', label='Class 2')
    ax.scatter(X[Y == -1, 0], X[Y == -1, 1], s=100, marker='x', label='Unlabeled')

    g1 = Ellipse(xy=m1, width=w1[1] * 3, height=w1[0] * 3, fill=False, linestyle='dashed', angle=a1, color='black',
                 linewidth=1)
    g1_1 = Ellipse(xy=m1, width=w1[1] * 2, height=w1[0] * 2, fill=False, linestyle='dashed', angle=a1, color='black',
                   linewidth=2)
    g1_2 = Ellipse(xy=m1, width=w1[1] * 1.4, height=w1[0] * 1.4, fill=False, linestyle='dashed', angle=a1,
                   color='black', linewidth=3)

    g2 = Ellipse(xy=m2, width=w2[1] * 3, height=w2[0] * 3, fill=False, linestyle='dashed', angle=a2, color='black',
                 linewidth=1)
    g2_1 = Ellipse(xy=m2, width=w2[1] * 2, height=w2[0] * 2, fill=False, linestyle='dashed', angle=a2, color='black',
                   linewidth=2)
    g2_2 = Ellipse(xy=m2, width=w2[1] * 1.4, height=w2[0] * 1.4, fill=False, linestyle='dashed', angle=a2,
                   color='black', linewidth=3)

    ax.set_xlabel(r'$x_0$', fontsize=16)
    ax.set_ylabel(r'$x_1$', fontsize=16)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)

    ax.legend(fontsize=16)

    plt.show()