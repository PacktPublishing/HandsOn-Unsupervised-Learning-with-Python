import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# For reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    T = np.expand_dims(np.linspace(0.0, 10.0, num=100), axis=1)
    X = (T * np.random.uniform(1.0, 1.5, size=(100, 1))) + np.random.normal(0.0, 3.5, size=(100, 1))
    df = pd.DataFrame(np.concatenate([T, X], axis=1), columns=['t', 'x'])

    # Perform the linear regression
    lr = LinearRegression()
    lr.fit(T, X)

    # Print the equation
    print('x(t) = {0:.3f}t + {1:.3f}'.format(lr.coef_[0][0], lr.intercept_[0]))

    # Show the diagram
    sns.set(style="white", palette="muted", color_codes=True)
    ax = sns.lmplot(data=df, x='t', y='x', height=8)
    plt.show()