import math
import numpy as np
import pandas as pd
import sys


def add_ones(array):
    new_array = np.ones([array.shape[0], array.shape[1]+1])
    new_array[:, 1:] = array
    return new_array


# Counts the gradient of function
def gradient(x, y, theta):
    grad = sum((sigmoid(x, theta) - y) * x)
    return np.array([[el] for el in grad]) / len(y[:, 0])


# Performs the gradient descent
def gradient_descent(x, y, theta, alpha, num_iter):
    ls = np.zeros(num_iter)
    for i in range(num_iter):
        ls[i] = logistic_cost(x, y, theta)

        theta -= gradient(x, y, theta) * alpha / len(y)

    return theta, ls


# Counts the cost function
def logistic_cost(hypothesis, real_results):
    return -sum(sum(real_results * np.log(hypothesis) + (1 - real_results) * np.log(1 - hypothesis))) / hypothesis.shape[0]


def max_index(mass):
    maximum = 0
    index = 0
    for i in range(len(mass)):
        if mass[i] > maximum:
            maximum = mass[i]
            index = i
    return index


def mean(array):
    array = np.array(array)
    return sum(sum(array)) / (array.shape[0] * array.shape[1])


def mistakes(hypothesis, real_results):
    n = 0
    for i in range(hypothesis.shape[0]):
        if max_index(hypothesis[i, :]) == real_results[i]:
            n += 1
    return n / hypothesis.shape[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Counts the cost function with regularization
def reg_logistic_cost(hypothesis, real_results, weights_1, weights_2):
    return -sum(sum(real_results * np.log(hypothesis) + (1 - real_results) * np.log(1 - hypothesis))) / hypothesis.shape[0] + sum(sum(weights_1)) + sum(sum(weights_2))


# Neural network cost function.
def nn_cost(output_layer, actual_results):
    cost = 0
    for i in range(output_layer.shape[0]):
        cost += np.matmul(np.log(output_layer[i, :]), actual_results) - np.matmul(np.log(1 - output_layer[i, :]), (1 - actual_results))
    return cost


def variance(array):
    array = np.array(array)
    mean = sum(sum(array)) / (array.shape[0] * array.shape[1])
    return sum(sum((array - mean)**2)) / (array.shape[0] * array.shape[1])


def generate_weights(numbers):
    weights = []
    for el in numbers:
        weights.append(np.random.sample(el))
    return weights


class NN:
    def __init__(self, train_data, test_data, train_results, test_results,
                 weights):
        self.train_data = (train_data - mean(train_data)) / (variance(train_data)**0.5)
        self.test_data = (test_data - mean(test_data)) / (variance(test_data)**0.5)
        self.train_results = np.array([np.eye(10)[el[0], :] for el in train_results])
        self.real_test_results = test_results
        self.test_results = np.array([np.eye(10)[el[0], :] for el in test_results])
        self.weights = weights
        self.layers = []
        self.activations = []
        self.delta_weights = []
        self.layer_errors = []

    def forward_propagation(self, data):
        self.activations.clear()
        self.layers.clear()
        self.activations.append(data)
        # print(self.activations[0].shape)
        for i in range(1, len(self.weights)+1):
            self.layers.append(np.matmul(self.activations[i-1], self.weights[i-1]))
            if i == len(self.weights):
                self.activations.append(sigmoid(self.layers[i-1]))
                # print(self.activations[i].shape)
            else:
                self.activations.append(sigmoid(add_ones(self.layers[i-1])))
                # print(self.activations[i].shape)

    def mini_batch_back_propagation(self, epochs, alpha, lmbda, batch_size):
        # This is a "mini-batch" gradient descent algorithm. This means that on an iteration we calculate the error on
        # batches and after that we adjust the weights according to the error of this batch

        # We collect the information about the errors in order to see how the model is improving
        errors = np.zeros([epochs])
        for j in range(epochs):
            self.forward_propagation(self.test_data)

            sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
            # sys.stdout.write(self.hypothesis[0, :])
            sys.stdout.flush()

            # We propagate forward and then calculate the error on validation data
            # self.forward_propagation(self.validation_data)
            errors[j] = logistic_cost(self.activations[-1], self.test_results)

            # We collect the info about delta_weights
            self.delta_weights = [np.zeros(el.shape) for el in self.weights]

            self.forward_propagation(self.train_data)

            for i in range(0, int(self.train_data.shape[0]), batch_size):

                # On every iteration we calculate the errors in layers, that is why on every iteration we should
                # clear the information about the errors and set it to zero
                self.layer_errors = [0 for el in self.weights]

                # The output error(which is the last error in self.layer_errors) is calculated as the difference between
                # the hypothesis(self.activations[-1]) and the rights results(self.train_results)
                self.layer_errors[-1] = self.activations[-1][i:i+batch_size, :] - self.train_results[i:i+batch_size, :]

                for t in range(len(self.activations)-3, -1, -1):
                    if t == len(self.activations)-3:
                        _ = np.matmul(self.layer_errors[t+1], np.transpose(self.weights[t+1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i+batch_size]))

                    else:
                        _ = np.matmul(self.layer_errors[t+1][:, 1:], np.transpose(self.weights[t+1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i+batch_size]))

                for t in range(len(self.delta_weights)):
                    if t == len(self.delta_weights) - 1:
                        h = np.transpose(np.matmul(np.transpose(self.layer_errors[t]), self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + self.weights[t] / batch_size
                    else:
                        h = np.transpose(np.matmul(np.transpose(self.layer_errors[t][:, 1:]), self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + lmbda * self.weights[t] / batch_size

            for t in range(len(self.delta_weights)):
                self.weights[t] -= alpha * self.delta_weights[t] / 100

        return errors

    def classical_back_propagation(self, epochs, alpha, lmbda):
        # This is a "mini-batch" gradient descent algorithm. This means that on an iteration we calculate the error on
        # batches and after that we adjust the weights according to the error of this batch

        # We collect the information about the errors in order to see how the model is improving
        errors = np.zeros([epochs])
        for j in range(epochs):
            self.forward_propagation(self.test_data)

            sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
            # sys.stdout.write(self.hypothesis[0, :])
            sys.stdout.flush()

            # We propagate forward and then calculate the error on validation data
            errors[j] = logistic_cost(self.activations[-1], self.test_results)

            # We collect the info about delta_weights
            self.delta_weights = [np.zeros(el.shape) for el in self.weights]

            self.forward_propagation(self.train_data)

            # On every iteration we calculate the errors in layers, that is why on every iteration we should
            # clear the information about the errors and set it to zero
            self.layer_errors = [0 for el in self.weights]

            # The output error(which is the last error in self.layer_errors) is calculated as the difference between
            # the hypothesis(self.activations[-1]) and the rights results(self.train_results)
            self.layer_errors[-1] = self.activations[-1] - self.train_results

            # Here we compute
            for t in range(len(self.activations) - 3, -1, -1):
                if t == len(self.activations) - 3:
                    _ = np.matmul(self.layer_errors[t + 1], np.transpose(self.weights[t + 1]))
                    self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t]))

                else:
                    _ = np.matmul(self.layer_errors[t + 1][:, 1:], np.transpose(self.weights[t + 1]))
                    self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t]))

            for t in range(len(self.delta_weights)):
                if t == len(self.delta_weights) - 1:
                    h = np.transpose(np.matmul(np.transpose(self.layer_errors[t]), self.activations[t]))
                    self.delta_weights[t] += h + self.weights[t]
                else:
                    h = np.transpose(
                        np.matmul(np.transpose(self.layer_errors[t][:, 1:]), self.activations[t]))
                    self.delta_weights[t] += h + lmbda * self.weights[t]

            for t in range(len(self.delta_weights)):
                self.weights[t] -= alpha * self.delta_weights[t]

        self.forward_propagation(self.test_data)
        print(mistakes(self.activations[-1], self.real_test_results))
        return errors
