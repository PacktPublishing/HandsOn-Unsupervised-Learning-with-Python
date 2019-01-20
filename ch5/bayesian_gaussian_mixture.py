import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

from matplotlib.patches import Ellipse

# For reproducibility
np.random.seed(1000)


nb_samples = 500
nb_centers = 5


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, center_box=[-5, 5],
                      centers=nb_centers, random_state=1000)

    # Train the model with concentration 1000 and 0.1
    for c in (1000.0, 0.1):
        gm = BayesianGaussianMixture(n_components=5, weight_concentration_prior=c,
                                     max_iter=10000, random_state=1000)
        gm.fit(X)

        print('Weights: {}'.format(gm.weights_))

        Y_pred = gm.fit_predict(X)

        print((Y_pred == 0).sum())
        print((Y_pred == 1).sum())
        print((Y_pred == 2).sum())
        print((Y_pred == 3).sum())
        print((Y_pred == 4).sum())

        # Compute the parameters of the Gaussian mixture
        m1 = gm.means_[0]
        m2 = gm.means_[1]
        m3 = gm.means_[2]
        m4 = gm.means_[3]
        m5 = gm.means_[4]

        c1 = gm.covariances_[0]
        c2 = gm.covariances_[1]
        c3 = gm.covariances_[2]
        c4 = gm.covariances_[3]
        c5 = gm.covariances_[4]

        we1 = 1 + gm.weights_[0]
        we2 = 1 + gm.weights_[1]
        we3 = 1 + gm.weights_[2]
        we4 = 1 + gm.weights_[3]
        we5 = 1 + gm.weights_[4]

        w1, v1 = np.linalg.eigh(c1)
        w2, v2 = np.linalg.eigh(c2)
        w3, v3 = np.linalg.eigh(c3)
        w4, v4 = np.linalg.eigh(c4)
        w5, v5 = np.linalg.eigh(c5)

        nv1 = v1 / np.linalg.norm(v1)
        nv2 = v2 / np.linalg.norm(v2)
        nv3 = v3 / np.linalg.norm(v3)
        nv4 = v4 / np.linalg.norm(v4)
        nv5 = v5 / np.linalg.norm(v5)

        a1 = np.arccos(np.dot(nv1[:, 1], [1.0, 0.0]) / np.linalg.norm(nv1[:, 1])) * 180.0 / np.pi
        a2 = np.arccos(np.dot(nv2[:, 1], [1.0, 0.0]) / np.linalg.norm(nv2[:, 1])) * 180.0 / np.pi
        a3 = np.arccos(np.dot(nv3[:, 1], [1.0, 0.0]) / np.linalg.norm(nv3[:, 1])) * 180.0 / np.pi
        a4 = np.arccos(np.dot(nv4[:, 1], [1.0, 0.0]) / np.linalg.norm(nv4[:, 1])) * 180.0 / np.pi
        a5 = np.arccos(np.dot(nv5[:, 1], [1.0, 0.0]) / np.linalg.norm(nv5[:, 1])) * 180.0 / np.pi

        # Show the results
        sns.set()

        fig, ax = plt.subplots(figsize=(22, 12))

        ax.scatter(X[Y_pred == 0, 0], X[Y_pred == 0, 1], s=80, marker='x', label='Gaussian 1')
        ax.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], s=80, marker='o', label='Gaussian 2')
        ax.scatter(X[Y_pred == 2, 0], X[Y_pred == 2, 1], s=80, marker='d', label='Gaussian 3')
        ax.scatter(X[Y_pred == 3, 0], X[Y_pred == 3, 1], s=80, marker='s', label='Gaussian 4')
        if c == 1000:
            ax.scatter(X[Y_pred == 4, 0], X[Y_pred == 4, 1], s=80, marker='^', label='Gaussian 5')

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

        g3 = Ellipse(xy=m3, width=w3[1] * 3, height=w3[0] * 3, fill=False, linestyle='dashed', angle=a3, color='black',
                     linewidth=1)
        g3_1 = Ellipse(xy=m3, width=w3[1] * 2, height=w3[0] * 2, fill=False, linestyle='dashed', angle=a3, color='black',
                       linewidth=2)
        g3_2 = Ellipse(xy=m3, width=w3[1] * 1.4, height=w3[0] * 1.4, fill=False, linestyle='dashed', angle=a3,
                       color='black', linewidth=3)

        g4 = Ellipse(xy=m4, width=w4[1] * 3, height=w4[0] * 3, fill=False, linestyle='dashed', angle=a4, color='black',
                     linewidth=1)
        g4_1 = Ellipse(xy=m4, width=w4[1] * 2, height=w4[0] * 2, fill=False, linestyle='dashed', angle=a4, color='black',
                       linewidth=2)
        g4_2 = Ellipse(xy=m4, width=w4[1] * 1.4, height=w4[0] * 1.4, fill=False, linestyle='dashed', angle=a4,
                       color='black', linewidth=3)

        ax.set_xlabel(r'$x_0$', fontsize=16)
        ax.set_ylabel(r'$x_1$', fontsize=16)

        ax.add_artist(g1)
        ax.add_artist(g1_1)
        ax.add_artist(g1_2)
        ax.add_artist(g2)
        ax.add_artist(g2_1)
        ax.add_artist(g2_2)
        ax.add_artist(g3)
        ax.add_artist(g3_1)
        ax.add_artist(g3_2)
        ax.add_artist(g4)
        ax.add_artist(g4_1)
        ax.add_artist(g4_2)

        if c == 1000:
            g5 = Ellipse(xy=m5, width=w5[1] * 3, height=w5[0] * 3, fill=False, linestyle='dashed', angle=a5,
                         color='black',
                         linewidth=1)
            g5_1 = Ellipse(xy=m5, width=w5[1] * 2, height=w5[0] * 2, fill=False, linestyle='dashed', angle=a5,
                           color='black',
                           linewidth=2)
            g5_2 = Ellipse(xy=m5, width=w5[1] * 1.4, height=w5[0] * 1.4, fill=False, linestyle='dashed', angle=a5,
                           color='black', linewidth=3)

            ax.add_artist(g5)
            ax.add_artist(g5_1)
            ax.add_artist(g5_2)

        ax.legend(fontsize=16)

        plt.show()


