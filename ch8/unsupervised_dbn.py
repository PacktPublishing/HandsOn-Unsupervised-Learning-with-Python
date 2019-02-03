import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# To install the DBN package: pip install git+git://github.com/albertbup/deep-belief-network.git
# Further information: https://github.com/albertbup/deep-belief-network
from dbn import UnsupervisedDBN

from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.utils import shuffle


# Set random seed for reproducibility
np.random.seed(1000)


nb_samples = 500


if __name__ == '__main__':
    # Load the dataset
    digits = load_digits()
    X_train = digits['data'] / np.max(digits['data'])
    Y_train = digits['target']

    X_train, Y_train = shuffle(X_train, Y_train, random_state=1000)
    X_train = X_train[0:nb_samples]
    Y_train = Y_train[0:nb_samples]

    # Train the unsupervised DBN
    unsupervised_dbn = UnsupervisedDBN(hidden_layers_structure=[32, 32, 16],
                                       learning_rate_rbm=0.025,
                                       n_epochs_rbm=500,
                                       batch_size=16,
                                       activation_function='sigmoid')

    X_dbn = unsupervised_dbn.fit_transform(X_train)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=1000)
    X_tsne = tsne.fit_transform(X_dbn)

    # Show the result
    fig, ax = plt.subplots(figsize=(22, 14))
    sns.set()

    markers = ['o', 'd', 'x', '^', 'v', '<', '>', 'P', 's', 'p']

    for i in range(10):
        ax.scatter(X_tsne[Y_train == i, 0], X_tsne[Y_train == i, 1], marker=markers[i], s=150,
                   label='Class {}'.format(i + 1))

    ax.set_xlabel(r'$x_0$', fontsize=16)
    ax.set_ylabel(r'$x_1$', fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=16)

    plt.show()