import numpy as np
import pandas as pd
import sys
from tools import *
# from matplotlib import pyplot as plt

if __name__ == '__main__':
    b = pd.read_csv('/Users/vladikturin/Desktop/pycharm_projects/recognition/train.csv')
    c = pd.read_csv('/Users/vladikturin/Desktop/pycharm_projects/recognition/test.csv')
    c = np.array(c)
    c = (c - mean(c)) / (variance(c) ** 0.5)
    c = add_ones(c)
    # We take the label values and put it into a separate matrix
    results = np.array(b['label']).reshape((42000, 1))
    # Now we do not need label values, and now we substitute it we bias unit
    b['label'] = pd.Series([1 for i in range(len(b['label']))])
    b = np.array(b)

    # At this step we standardise the input data so that values are between 0 and 1
    b = (b - mean(b)) / (variance(b) ** 0.5)

    train_data = b[0:int(0.8 * b.shape[0]), :]
    test_data = b[int(0.8 * b.shape[0]):, :]

    train_results = results[0:int(0.8 * results.shape[0]), :]
    test_results = results[int(0.8 * results.shape[0]):, :]

    # weights = generate_weights([[785, 300], [301, 100], [101, 10]])
    # weights = generate_weights([[785, 300], [301, 150], [151, 70], [71, 10]])
    weights = generate_weights([[785, 100], [101, 10]])
    a = NN(train_data, test_data, train_results, test_results, weights)
    # a.mini_batch_back_propagation(10000, 0.001, 200, 100)

    a.get_result(0.001, 50, 100, 0.8)

    # Now we are going to check our results on another test set
    a.submit(c)

