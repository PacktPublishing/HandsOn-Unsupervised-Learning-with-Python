import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import poisson


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the initial observation set
    obs = np.array([7, 11, 9, 9, 8, 11, 9, 9, 8, 7, 11, 8, 9, 9, 11, 7, 10, 9, 10, 9, 7, 8, 9, 10, 13])
    mu = np.mean(obs)

    print('mu = {}'.format(mu))

    # Show the distribution
    sns.set(style="white", palette="muted", color_codes=True)
    fig, ax = plt.subplots(figsize=(14, 7), frameon=False)

    sns.distplot(obs, kde=True, color="b", ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Print some probabilities
    print('P(more than 8 trains) = {}'.format(poisson.sf(8, mu)))
    print('P(more than 9 trains) = {}'.format(poisson.sf(9, mu)))
    print('P(more than 10 trains) = {}'.format(poisson.sf(10, mu)))
    print('P(more than 11 trains) = {}'.format(poisson.sf(11, mu)))

    # Add new observations
    new_obs = np.array([13, 14, 11, 10, 11, 13, 13, 9, 11, 14, 12, 11, 12,
                        14, 8, 13, 10, 14, 12, 13, 10, 9, 14, 13, 11, 14, 13, 14])

    obs = np.concatenate([obs, new_obs])
    mu = np.mean(obs)

    print('mu = {}'.format(mu))

    # Repeat the analysis of the same probabilities
    print('P(more than 8 trains) = {}'.format(poisson.sf(8, mu)))
    print('P(more than 9 trains) = {}'.format(poisson.sf(9, mu)))
    print('P(more than 10 trains) = {}'.format(poisson.sf(10, mu)))
    print('P(more than 11 trains) = {}'.format(poisson.sf(11, mu)))

    # Generate 2000 samples from the Poisson process
    syn = poisson.rvs(mu, size=2000)

    # Plot the complete distribution
    fig, ax = plt.subplots(figsize=(14, 7), frameon=False)

    sns.distplot(syn, kde=True, color="b", ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

