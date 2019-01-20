import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Generate the dataset
    nb_samples = [1000, 800, 500, 380, 280, 150, 120, 100, 50, 30]

    ages = []

    for n in nb_samples:
        i = np.random.uniform(10, 80, size=2)
        a = np.random.uniform(i[0], i[1], size=n).astype(np.int32)
        ages.append(a)

    ages = np.concatenate(ages)

    # Compute the histogram
    # Any FutureWarning is related to SciPy deprecations which are still employed by NumPy but
    # it doesn't affect the results
    h, e = np.histogram(ages, bins='auto')

    print('Histograms counts: {}'.format(h))
    print('Bin edges: {}'.format(e))

    # Show the histogram
    sns.set()

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.distplot(ages, kde=False, ax=ax, label='Age count')

    ax.set_xlabel('Age', fontsize=14)
    ax.set_ylabel('Number of entries', fontsize=14)

    ax.set_xticks(e)

    ax.legend()

    plt.show()

    # Compute the probability for a sample interval
    d = e[1] - e[0]
    p50 = float(h[12]) / float(ages.shape[0])

    print('P(48.84 < x < 51.58) = {:.2f} ({:.2f}%)'.format(p50, p50 * 100.0))