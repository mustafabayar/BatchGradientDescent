from random import randrange
import pandas as pd
import numpy as np
import GradientDescent as gd


def cross_validation_split(dataset, folds=5):
    dataset_split = list()
    dataset_copy = dataset.values.tolist()
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            # Instead of shuffling the data before processing,
            # creating splits by taking rows from random indexes is another option
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def perform_gradient_on_splits(splits, learning_rate, iterations):
    total_mae = 0
    for i in range(len(splits)):
        split = splits[:]
        test_data = split[i]
        split.remove(test_data)
        train_data = []

        for x in split:
            train_data.extend(x)

        # ---------- TRAINING SPLIT ---------- #

        df_train = pd.DataFrame(train_data)

        # Target Variable
        y_train = df_train.iloc[:, -1]
        y_train = np.matrix(y_train).T

        # Features
        X_train = df_train.iloc[:, :-1]
        X_train = (X_train - X_train.mean()) / X_train.std()
        ones = np.ones([X_train.shape[0], 1])
        X_train = np.concatenate((ones, X_train), axis=1)
        X_train = np.matrix(X_train)

        theta = np.zeros([1, X_train.shape[1]])

        result = gd.gradient_descent(X_train, y_train, theta, learning_rate, iterations)
        #plot_graph(iterations, result)
        final_theta = result[0]


        # ---------- TEST SPLIT ---------- #

        df_test = pd.DataFrame(test_data)

        # Target Variable
        y_test = df_test.iloc[:, -1]
        y_test = np.matrix(y_test).T

        # Features
        X_test = df_test.iloc[:, :-1]
        X_test = (X_test - X_test.mean()) / X_test.std()
        ones = np.ones([X_test.shape[0], 1])
        X_test = np.concatenate((ones, X_test), axis=1)
        X_test = np.matrix(X_test)

        final_predictions = X_test.dot(final_theta.T)
        mae = gd.mean_absolute_error(final_predictions, y_test.T)
        total_mae = total_mae + mae
        print("Mean Absolute Error of Test Set {0}: {1}".format(i + 1, mae))

    print("Average of MAE: {0}".format(total_mae / 5))
