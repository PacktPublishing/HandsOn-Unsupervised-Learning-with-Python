import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KernelDensity


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

    # Train KDE with Gaussian kernels and 3 bandwidths
    kd_01 = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kd_05 = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kd_15 = KernelDensity(kernel='gaussian', bandwidth=1.5)

    kd_01.fit(ages.reshape(-1, 1))
    kd_05.fit(ages.reshape(-1, 1))
    kd_15.fit(ages.reshape(-1, 1))

    # Show the results
    sns.set()

    fig, ax = plt.subplots(3, 1, figsize=(14, 20), sharex=True)

    data = np.arange(10, 70, 0.05).reshape(-1, 1)

    ax[0].plot(data, np.exp(kd_01.score_samples(data)))
    ax[0].set_title('Bandwidth = 0.1', fontsize=14)
    ax[0].set_ylabel('Density', fontsize=14)

    ax[1].plot(data, np.exp(kd_05.score_samples(data)))
    ax[1].set_title('Bandwidth = 0.5', fontsize=14)
    ax[1].set_ylabel('Density', fontsize=14)

    ax[2].plot(data, np.exp(kd_15.score_samples(data)))
    ax[2].set_title('Bandwidth = 1.5', fontsize=14)
    ax[2].set_xlabel('Age', fontsize=14)
    ax[2].set_ylabel('Density', fontsize=14)

    plt.show()

    # Compute the optimal bandwidth (method 1)
    N = float(ages.shape[0])
    h = 1.06 * np.std(ages) * np.power(N, -0.2)

    print('h = {:.3f}'.format(h))

    # Compute the optimal bandwidth (method 2)
    IQR = np.percentile(ages, 75) - np.percentile(ages, 25)
    h = 0.9 * np.min([np.std(ages), IQR / 1.34]) * np.power(N, -0.2)

    print('h = {:.3f}'.format(h))

    # Train KDE with different kernels and bandwidth = 2.0
    kd_gaussian = KernelDensity(kernel='gaussian', bandwidth=2.0)
    kd_epanechnikov = KernelDensity(kernel='epanechnikov', bandwidth=2.0)
    kd_exponential = KernelDensity(kernel='exponential', bandwidth=2.0)

    kd_gaussian.fit(ages.reshape(-1, 1))
    kd_epanechnikov.fit(ages.reshape(-1, 1))
    kd_exponential.fit(ages.reshape(-1, 1))

    # Show the results
    fig, ax = plt.subplots(3, 1, figsize=(14, 20), sharex=False)

    data = np.arange(10, 70, 0.05).reshape(-1, 1)

    ax[0].plot(data, np.exp(kd_gaussian.score_samples(data)))
    ax[0].set_title('Gaussian Kernel', fontsize=14)
    ax[0].set_ylabel('Density', fontsize=14)

    ax[1].plot(data, np.exp(kd_epanechnikov.score_samples(data)))
    ax[1].set_title('Epanechnikov Kernel', fontsize=14)
    ax[1].set_ylabel('Density', fontsize=14)
    ax[1].set_xlabel('Age', fontsize=14)

    ax[2].plot(data, np.exp(kd_exponential.score_samples(data)))
    ax[2].set_title('Exponential Kernel', fontsize=14)
    ax[2].set_xlabel('Age', fontsize=14)
    ax[2].set_ylabel('Density', fontsize=14)

    plt.show()

    # Perform a sample anomaly detection
    test_data = np.array([12, 15, 18, 20, 25, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]).reshape(-1, 1)

    test_densities_epanechnikov = np.exp(kd_epanechnikov.score_samples(test_data))
    test_densities_gaussian = np.exp(kd_gaussian.score_samples(test_data))

    for age, density in zip(np.squeeze(test_data), test_densities_epanechnikov):
        print('p(Age = {:d}) = {:.7f} ({})'.format(age, density, 'Anomaly' if density < 0.005 else 'Normal'))



