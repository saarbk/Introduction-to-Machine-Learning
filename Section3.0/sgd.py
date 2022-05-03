#################################
# Your name: Saar Barak
#################################

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import numpy.random
import scipy
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:10000], :].astype(float)
    train_labels = (labels[train_idx[:10000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[1000:], :].astype(float)
    validation_labels = (labels[train_idx[1000:]] == pos) * 2 - 1

    test_data_unscaled = data[1000 + test_idx, :].astype(float)
    test_labels = (labels[1000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w_t = np.zeros(data[1].shape)
    for t in range(T - 1):
        r = np.random.randint(0, high=data.shape[0], dtype=int)
        eta_t = eta_0 / (t + 1)
        decision = labels[r] * np.dot(w_t, data[r]) < 1
        w_t = w_t * (1 - eta_t)
        if decision:
            w_t += eta_t * C * labels[r] * data[r]

    return w_t


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        w = w.reshape(data.shape[1], )
        i = np.random.randint(0, high=data.shape[0], dtype=int)
        gradient_w_with_f_i = log_loss_gradient_calculator(w, data[i], labels[
            i])  # scipy.special.softmax(-labels[i] * np.dot(w, data[i])) #
        w = np.add(w, (eta_0 / t) * (np.dot(gradient_w_with_f_i, data[i])))
    return w
    pass

    #################################

    # Place for additional code

    #################################


def loss_accuracy(data, labels, w):
    loss = 0
    for i in range(data.shape[0]):
        y = 1 if np.dot(w, data[i]) >= 0 else -1
        if y != labels[i]:
            loss += 1
    return 1 - (loss / data.shape[0])


def SGD_hinge_for_eta():
    eta_to_accuracies = {}
    Eta = np.geomspace(10 ** (-5), 1000, 10, endpoint=True)
    for eta in Eta:
        eta_to_accuracies[eta] = []
        for i in range(10):
            w_i = (SGD_hinge(train_data[-1000:], train_labels[-1000:], 1, eta, 1000))
            eta_to_accuracies[eta].append(hinge_loss_accuracy(test_data[-100:], test_labels[-100:], w_i))
    return eta_to_accuracies


def find_best_eta_zero(train_data, train_labels, validation_data, validation_labels, q):
    choices = [np.float_power(10, k) for k in range(-5, 4)]
    average_accuracies_list = []
    for i in choices:
        accuracies_list = []
        for j in range(10):
            if q == 1:
                w = SGD_hinge(train_data, train_labels, 1, i, 1000)
            else:
                w = SGD_log(train_data, train_labels, i, 1000)
            accuracy = hinge_loss_accuracy_calculator(validation_data, validation_labels, w)
            accuracies_list.append(accuracy)
        average_accuracies_list.append(np.average(accuracies_list))

    plt.plot(choices, average_accuracies_list)
    plt.show()

def find_best_c(train_data, train_labels, validation_data, validation_labels):
    choices = [np.float_power(10, k) for k in range(-5, 6)]
    average_accuracies_list = []
    for i in choices:
        accuracies_list = []
        for j in range(10):
            w = SGD_hinge(train_data, train_labels, i, 1, 1000)
            accuracy = hinge_loss_accuracy_calculator(validation_data, validation_labels, w)
            accuracies_list.append(accuracy)
        average_accuracies_list.append(np.average(accuracies_list))

    plt.plot(choices, average_accuracies_list, color='G')
    plt.ylabel('average accuracy of 10 samples')
    plt.show()


def find_accuracy_on_test_data(train_data, train_labels, test_data, test_labels, q):
    if q == 1:
        w = SGD_hinge(train_data, train_labels, 10 ** (-4), 1, 20000)
    else:
        w = SGD_log(train_data, train_labels, 100, 20000)
    return hinge_loss_accuracy_calculator(test_data, test_labels, w)


def show_w(data, labels, q):
    if q == 1:
        w = SGD_hinge(data, labels, 10 ** (-4), 1, 20000)
    else:
        w = SGD_log(data, labels, 100, 20000)

    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()


def log_loss_gradient_calculator(w, x, y):
    exponent = scipy.special.softmax(-y * np.dot(w, x))
    return (exponent * ((-y)) / (1 + exponent)) * x


def plot_q1(eta_choices):
    for eta in eta_choices:
        accuracies = eta_choices[eta]
        plt.scatter(eta_choices[eta], accuracies)
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(eta_choices.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(eta_choices.items())])
    plt.errorbar(sorted(eta_choices.items()), accuracies_mean, yerr=accuracies_std)
    plt.semilogx(np.geomspace(0.00001, 10000, 10, endpoint=True), accuracies_mean, 'o')
    plt.semilogx(np.geomspace(0.00001, 10000, 10, endpoint=False), t[1], 'o')
    plt.axis([0.00001, 100000, 0, 1])
    plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
    plt.show()
    pass


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
