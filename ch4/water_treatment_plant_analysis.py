import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet


# For reproducibility
np.random.seed(1000)


# Download the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/
# and set the path to .data file
data_path = '<YOUR_PATH>/water-treatment.data'


nb_clusters = [4, 6, 8, 10]
linkages = ['single', 'complete', 'ward', 'average']


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv(data_path, header=None, index_col=0, na_values='?').astype(np.float64)
    df.fillna(df.mean(), inplace=True)

    # Standardize the dataset
    ss = StandardScaler(with_std=False)
    sdf = ss.fit_transform(df)

    # Perform the TSNE non-linear dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=10, random_state=1000)
    data_tsne = tsne.fit_transform(sdf)

    df_tsne = pd.DataFrame(data_tsne, columns=['x', 'y'], index=df.index)
    dff = pd.concat([df, df_tsne], axis=1)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 11))

    with sns.plotting_context("notebook", font_scale=1.5):
        sns.scatterplot(x='x',
                        y='y',
                        size=0,
                        sizes=(120, 120),
                        data=dff,
                        legend=False,
                        ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()

    # Analyze the result of different linkages and number of clusters
    cpcs = np.zeros(shape=(len(linkages), len(nb_clusters)))
    silhouette_scores = np.zeros(shape=(len(linkages), len(nb_clusters)))

    for i, l in enumerate(linkages):
        for j, nbc in enumerate(nb_clusters):
            dm = pdist(sdf, metric='minkowski', p=2)
            Z = linkage(dm, method=l)
            cpc, _ = cophenet(Z, dm)
            cpcs[i, j] = cpc

            ag = AgglomerativeClustering(n_clusters=nbc, affinity='euclidean', linkage=l)
            Y_pred = ag.fit_predict(sdf)
            sls = silhouette_score(sdf, Y_pred, random_state=1000)
            silhouette_scores[i, j] = sls

    fig, ax = plt.subplots(len(nb_clusters), 2, figsize=(20, 20), sharex=True)

    for i in range(len(nb_clusters)):
        ax[i, 0].plot(cpcs[:, i])
        ax[i, 0].set_ylabel('Cophenetic correlation', fontsize=14)
        ax[i, 0].set_title('Number of clusters: {}'.format(nb_clusters[i]), fontsize=14)

        ax[i, 1].plot(silhouette_scores[:, i])
        ax[i, 1].set_ylabel('Silhouette score', fontsize=14)
        ax[i, 1].set_title('Number of clusters: {}'.format(nb_clusters[i]), fontsize=14)

    plt.xticks(np.arange(len(linkages)), linkages)

    plt.show()

    # Show the truncated dendrogram for a complete linkage
    dm = pdist(sdf, metric='euclidean')
    Z = linkage(dm, method='complete')

    fig, ax = plt.subplots(figsize=(25, 20))

    d = dendrogram(Z, orientation='right', truncate_mode='lastp', p=80, no_labels=True, ax=ax)

    ax.set_xlabel('Dissimilarity', fontsize=18)
    ax.set_ylabel('Samples (80 leaves)', fontsize=18)

    plt.show()

    # Perform the clustering (both with 8 and 2 clusters)
    for n in (8, 2):
        ag = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='complete')
        Y_pred = ag.fit_predict(sdf)

        df_pred = pd.Series(Y_pred, name='Cluster', index=df.index)
        pdff = pd.concat([dff, df_pred], axis=1)

        # Show the results of the clustering
        fig, ax = plt.subplots(figsize=(18, 11))

        with sns.plotting_context("notebook", font_scale=1.5):
            sns.scatterplot(x='x',
                            y='y',
                            hue='Cluster',
                            size='Cluster',
                            sizes=(120, 120),
                            palette=sns.color_palette("husl", n),
                            data=pdff,
                            ax=ax)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        plt.show()









