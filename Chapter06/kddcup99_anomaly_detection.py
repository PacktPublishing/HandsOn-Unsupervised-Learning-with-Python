import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_kddcup99
from sklearn.neighbors import KernelDensity


# For reproducibility
np.random.seed(1000)


def is_anomaly(kd, source, destination, medium_thr=0.03, high_thr=0.015):
    xs = np.log(source + 0.1)
    xd = np.log(destination + 0.1)
    data = np.array([[xs, xd]])

    density = np.exp(kd.score_samples(data))[0]

    if density >= medium_thr:
        return density, 'Normal connection'
    elif density >= high_thr:
        return density, 'Medium risk'
    else:
        return density, 'High risk'


if __name__ == '__main__':
    # Load the dataset
    kddcup99 = fetch_kddcup99(subset='http', percent10=True, random_state=1000)

    X = kddcup99['data'].astype(np.float64)
    Y = kddcup99['target']

    print('Statuses: {}'.format(np.unique(Y)))
    print('Normal samples: {}'.format(X[Y == b'normal.'].shape[0]))
    print('Anomalies: {}'.format(X[Y != b'normal.'].shape[0]))

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    IQRs = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)

    # Show the histogram of the durations
    # Any FutureWarning is related to SciPy deprecations which are still employed by NumPy but
    # it doesn't affect the results
    h0, e0 = np.histogram(X[:, 0], bins='auto')

    sns.set()

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.distplot(X[:, 0], kde=False, ax=ax)

    ax.set_xlabel('Duration', fontsize=14)
    ax.set_ylabel('Number of entries', fontsize=14)

    ax.set_xticks(e0)

    plt.show()

    # Compute the optimal bandwidth
    N = float(X.shape[0])

    h0 = 0.9 * np.min([stds[0], IQRs[0] / 1.34]) * np.power(N, -0.2)
    h1 = 0.9 * np.min([stds[1], IQRs[1] / 1.34]) * np.power(N, -0.2)
    h2 = 0.9 * np.min([stds[2], IQRs[2] / 1.34]) * np.power(N, -0.2)

    print('h0 = {:.3f}, h1 = {:.3f}, h2 = {:.3f}'.format(h0, h1, h2))

    # Show the KDE for normal and malicious connections
    fig, ax = plt.subplots(2, 3, figsize=(22, 10))

    sns.distplot(X[Y == b'normal.', 0], kde=True, ax=ax[0, 0], label='KDE')
    sns.distplot(X[Y == b'normal.', 1], kde=True, ax=ax[0, 1], label='KDE')
    sns.distplot(X[Y == b'normal.', 2], kde=True, ax=ax[0, 2], label='KDE')

    sns.distplot(X[Y != b'normal.', 0], kde=True, ax=ax[1, 0], label='KDE')
    sns.distplot(X[Y != b'normal.', 1], kde=True, ax=ax[1, 1], label='KDE')
    sns.distplot(X[Y != b'normal.', 2], kde=True, ax=ax[1, 2], label='KDE')

    ax[0, 0].set_title('Duration', fontsize=16)
    ax[0, 1].set_title('Source bytes', fontsize=16)
    ax[0, 2].set_title('Destination bytes', fontsize=16)

    ax[0, 0].set_xticks(np.arange(-4, 12, 2))
    ax[1, 0].set_xticks(np.arange(-4, 12, 2))

    ax[0, 1].set_xticks(np.arange(-10, 16, 2))
    ax[1, 1].set_xticks(np.arange(-10, 16, 2))

    ax[0, 2].set_xticks(np.arange(-2, 14, 2))
    ax[1, 2].set_xticks(np.arange(-2, 14, 2))

    plt.show()

    # Perform the KDE
    X = X[:, 1:]

    kd = KernelDensity(kernel='gaussian', bandwidth=0.025)
    kd.fit(X[Y == b'normal.'])

    Yn = np.exp(kd.score_samples(X[Y == b'normal.']))
    Ya = np.exp(kd.score_samples(X[Y != b'normal.']))

    print('Mean normal: {:.5f} - Std: {:.5f}'.format(np.mean(Yn), np.std(Yn)))
    print('Mean anomalies: {:.5f} - Std: {:.5f}'.format(np.mean(Ya), np.std(Ya)))

    print(np.sum(Yn < 0.05))
    print(np.sum(Yn < 0.03))
    print(np.sum(Yn < 0.02))
    print(np.sum(Yn < 0.015))

    print(np.sum(Ya < 0.015))

    # Perform some sample anomaly detections
    print('p = {:.2f} - {}'.format(*is_anomaly(kd, 200, 1100)))
    print('p = {:.2f} - {}'.format(*is_anomaly(kd, 360, 200)))
    print('p = {:.2f} - {}'.format(*is_anomaly(kd, 800, 1800)))

    # Show the bivariate KDE plot
    fig, ax = plt.subplots(figsize=(13, 8))

    sns.kdeplot(X[Y != b'normal.', 0], X[Y != b'normal.', 1], cmap="Reds", shade=True, shade_lowest=False, kernel='gau',
                bw=0.025, ax=ax, label='Anomaly')
    sns.kdeplot(X[Y == b'normal.', 0], X[Y == b'normal.', 1], cmap="Blues", shade=True, shade_lowest=False,
                kernel='gau', bw=0.025, ax=ax, label='Normal')

    ax.set_xlabel('Source Bytes (logarithmic)', fontsize=14)
    ax.set_ylabel('Destination Bytes (logarithmic)', fontsize=14)

    ax.set_xlim(4, 12)
    ax.set_ylim(5, 11)

    ax.legend()

    plt.show()