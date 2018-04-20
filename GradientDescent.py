import numpy as np
import matplotlib.pyplot as plt


def mean_absolute_error(predictions, real_values):
    output_errors = np.average(np.abs(predictions - real_values))
    return output_errors


def plot_graph(iterations, cost):
    iteration = np.matrix([i for i in range(iterations)])
    plt.plot(iteration.T, cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs Training')
    plt.show()


def compute_cost(X, y, theta):
    m = y.size
    prediction = X.dot(theta.T)
    error = np.power((prediction - y), 2)
    cost = np.sum(error) / (2 * m)
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    cost = np.zeros(shape=iterations)
    number_of_features = theta.size
    for i in range(iterations):
        prediction = X.dot(theta.T)
        error = prediction - y
        for j in range(number_of_features):
            temp = np.multiply(error, X[:, j])
            theta[0, j] = theta[0, j] - ((learning_rate / len(X)) * np.sum(temp))
        cost[i] = compute_cost(X, y, theta)
    return theta, cost

