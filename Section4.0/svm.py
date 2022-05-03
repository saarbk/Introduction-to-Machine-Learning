import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import random

# plt.rcParams['text.usetex'] = True

def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100

# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1, X2], axis=1)
y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])
"""
             Q-1
"""
data_point = np.ndarray(shape=(200, 2))
names = [r"\bf{Polynomial Degree $2$}", r"\bf{Polynomial Degree $3$}"]
x1 = np.concatenate((X[50:150], np.concatenate((X[:50], X[-50:]))))
y1 = np.concatenate((y[50:150], np.concatenate((y[:50], y[-50:]))))

result = [svm.SVC(C=10, kernel='poly', degree=2).fit(x1[:100], y1[:100]),
          svm.SVC(C=10, kernel='poly', degree=3).fit(x1[-100:], y1[-100:])]
plot_results(result, names, x1, y1)

"""
             Q-2
"""
result2 = [svm.SVC(C=10, kernel='poly', degree=2, coef0=0.01).fit(x1[:100], y1[:100]),
           svm.SVC(C=10, kernel='poly', degree=3, coef0=0.01).fit(x1[-100:], y1[-100:])]

plot_results(result2, names, x1, y1)
"""
             Q-3
"""
names = [r"\bf{Polynomial Degree $2$}", r"\bf{ RBF kernel $\gamma = 10$}"]
y2 = np.concatenate((y[:100], [1 if random.random() < 0.1 else -1 for t in range(100)]))
y2 = np.concatenate((y2[50:150], np.concatenate((y2[:50], y2[-50:]))))
result3 = [svm.SVC(C=10, kernel='poly', degree=2).fit(x1[:100], y2[:100]),
           svm.SVC(kernel='rbf', gamma=10).fit(x1[-100:], y2[-100:])]
plot_results(result3, names, x1, y2)

names = [r"kernel $\gamma = 0.5$", r"kernel $\gamma = 5$", r"kernel $\gamma = 25$",
         r" kernel $\gamma = 100$"]
result3 = [svm.SVC(C=10, kernel='rbf', gamma=0.5).fit(x1[:100], y2[:100]),
           svm.SVC(C=10, kernel='rbf', gamma=5).fit(x1[-100:], y2[-100:]),
           svm.SVC(C=10, kernel='rbf', gamma=25).fit(x1[:100], y2[:100]),
           svm.SVC(C=10, kernel='rbf', gamma=100).fit(x1[-100:], y2[-100:])
           ]
plot_results(result3, names, x1, y2)
