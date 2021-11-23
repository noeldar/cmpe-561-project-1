# Example of calculating class probabilities
import numpy as np
import pandas as pd
from math import sqrt
from math import pi
from math import exp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import operator




# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent



def calculate_statistics(X, y):
    mean = X.groupby(y).apply(np.mean)
    stds = X.groupby(y).apply(np.std)
    return mean,stds

def prior_probabilities(X, y):
    class_sizes = X.groupby(y).apply(lambda x: len(x))
    probs={}
    j=0
    for i in class_sizes:
        probs[j]=float(i/len(X))
        j=j+1

    return probs

def predict(X_test, X_train, y_train):
    class_prior_probs= prior_probabilities(X_train, y_train)
    mean, std = calculate_statistics(X_train, y_train)

    y_pred =[]
    for i in range(len(X_test)):
        prob_pred={}
        for j in list(set(y_train)):
            prob_pred[j]=class_prior_probs[j]
            for index, feat in enumerate(X_test.iloc[i]):
                #print(feat)
                #print(mean.iloc[j, index])
                #print(std.iloc[j, index])
                prob_pred[j] *= calculate_probability(feat, mean.iloc[j, index], std.iloc[j, index])
                #print(prob_pred[j])
                #print("****************************************************************")

        inverse = [(value, key) for key, value in prob_pred.items()]
        y_pred.append(max(inverse)[1])

    return y_pred

data =load_iris()
X, y, column_names = data['data'], data['target'], data['feature_names']
X = pd.DataFrame(X, columns=column_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#prior_probabilities(X_train, y_train)
print(predict(X_test, X_train, y_train))
