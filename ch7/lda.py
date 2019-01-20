import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    news = fetch_20newsgroups(subset='all',
                              categories=('rec.autos', 'comp.sys.mac.hardware'),
                              remove=('headers', 'footers', 'quotes'), random_state=1000)

    corpus = news['data']
    labels = news['target']

    # Vectorize the dataset
    cv = CountVectorizer(strip_accents='unicode', stop_words='english', analyzer='word', token_pattern='[a-z]+')
    Xc = cv.fit_transform(corpus)

    print(len(cv.vocabulary_))

    # Perform the LDA
    lda = LatentDirichletAllocation(n_components=2, learning_method='online', max_iter=100, random_state=1000)
    Xl = lda.fit_transform(Xc)

    # Show the top-10 words per topic
    Mwts_lda = np.argsort(lda.components_, axis=1)[::-1]

    for t in range(2):
        print('\nTopic ' + str(t))
        for i in range(10):
            print(cv.get_feature_names()[Mwts_lda[t, i]])

    # Show the sample messages
    print(corpus[100])
    print(corpus[200])

    # Show the topic mixtures
    print(Xl[100])
    print(Xl[200])

    # Show the mixtures for both sub-categories
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

    x0 = Xl[labels == 0]
    x1 = Xl[labels == 1]

    ax[0].scatter(x0[:, 0], x0[:, 1])
    ax[0].set_xlabel('Topic 0', fontsize=16)
    ax[0].set_ylabel('Topic 1', fontsize=16)
    ax[0].set_title('comp.sys.mac.hardware', fontsize=16)

    ax[1].scatter(x1[:, 0], x1[:, 1])
    ax[1].set_xlabel('Topic 0', fontsize=16)
    ax[1].set_title('rec.autos', fontsize=16)

    plt.show()

