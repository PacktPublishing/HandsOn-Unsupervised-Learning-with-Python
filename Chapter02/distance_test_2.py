import numpy as np

from scipy.spatial.distance import cdist


# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the distance matrix
    distances = []

    for i in range(1, 2500, 10):
        d = cdist(np.array([[0, 0]]), np.array([[5, float(i / 500)]]), metric='minkowski', p=15)[0][0]
        distances.append(d)

    print('Avg(distances) = {}'.format(np.mean(distances)))
    print('Std(distances) = {}'.format(np.std(distances)))



