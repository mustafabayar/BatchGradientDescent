import pandas as pd
import numpy as np
import GradientDescent as gd
import CrossValidation as cv

data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

important_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
important_features_test = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

data_set = data_train[important_features].copy()
data_set_test = data_test[important_features_test].copy()

# Find out which columns have null values
null_columns = data_set_test.columns[data_set_test.isnull().any(axis=0)]
# Find total count of those columns appeared as null in the set
null_list = data_set_test[null_columns].isnull().sum().sort_values(ascending=False)
# print("Missing Attributes: {0}".format(null_list))
data_set_test.loc[data_set_test['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 0
data_set_test.loc[data_set_test['GarageCars'].isnull(), 'GarageCars'] = 0

# Target Variable
y = data_set['SalePrice']
y = np.matrix(y).T

# Features
X = data_set.drop(columns='SalePrice')
X = (X - X.mean()) / X.std()
X.insert(0, "Intercept", 1)
X = np.matrix(X)

# Test Set
X_test = data_set_test
X_test = (X_test - X_test.mean()) / X_test.std()
X_test.insert(0, "Intercept", 1)
X_test = np.matrix(X_test)

# Initial Thetas
theta = np.matrix(np.zeros(shape=X.shape[1]))

# Parameters
learning_rate = 0.01
iteration = 500

print("\nRunning Linear Regression On Whole Set")
result = gd.gradient_descent(X, y, theta, learning_rate, iteration)
gd.plot_graph(iteration, result[1])

final_predictions = X.dot(result[0].T)
mae = gd.mean_absolute_error(final_predictions, y)
print("Mean Absolute Error: {0}".format(mae))

print("\nRunning Linear Regression On Split Sets")
splits = cv.cross_validation_split(data_set, 5)
cv.perform_gradient_on_splits(splits, learning_rate, iteration)

prediction_of_test_set = X_test.dot(result[0].T)
prediction_df = pd.DataFrame(prediction_of_test_set)
prediction_df.columns = ['SalePrice']

df_submission = pd.concat([data_test['Id'], prediction_df], axis=1)
df_submission.to_csv('data/Submission.csv', index=False)
