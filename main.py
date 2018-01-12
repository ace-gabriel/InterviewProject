import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# visualization

dataset = pd.read_csv("seattle.csv", sep = '\t').values

def plot_relation(a):

    """ A function that plots scatter plot of input a and rent
        Use this function to determine a possible correlation """


    rent = dataset[:,27]

    index = list(i for i in range(0, len(rent)) if rent[i] == '\\N' or pd.isnull(rent[i]))
    index2 = list(i for i in range(0, len(a)) if a[i] == '\\N' or pd.isnull(a[i]))

    a = np.delete(a, index + index2).astype(float)
    rent = np.delete(rent, index + index2).astype(float)

    plt.scatter(a, rent)
    plt.show()

def plot_room():
    # Special Handle of room type
    rtype = dataset[:,3]
    rent = dataset[:,27]

    index = list(i for i in range(0, len(rtype)) if not isinstance(rtype[i], str))
    index2 = list(i for i in range(0, len(rent)) if rent[i] == '\\N' or pd.isnull(rent[i]))

    rtype = np.delete(rtype, index + index2)
    rent = np.delete(rent, index + index2).astype(float)

    Encoder = LabelEncoder()
    rtype = Encoder.fit_transform(rtype).astype(float)

    plt.scatter(rtype, rent)
    plt.show()


def plot_other():
    beds = dataset[:,4]
    plot_relation(beds)
    size = dataset[:,6]
    plot_relation(size)
    score = dataset[:,14]
    plot_relation(score)
    adj_score = dataset[:,20]
    plot_relation(adj_score)
    p_score = dataset[:,21]
    plot_relation(p_score)
    n_score = dataset[:,22]
    plot_relation(n_score)

# preprocessing
# drop & clean data with N/A and nan

def main():
    x = []
    dataset_y = dataset[:,27]
    index_d = list(i for i in range(0, len(dataset_y)) if dataset_y[i] == '\\N' or pd.isnull(dataset_y[i]))

    for i in range(0, len(beds)):
        x.append(beds[i])
        x.append(size[i])

    x = [x[i: i + 2] for i in range(0, len(x), 2)]
    drop = []
    for i in range(0, len(x)):
        for j in range(0, 2):
            if x[i][j] == '\\N' or pd.isnull(x[i][j]):
                drop.append(i)

    drop = drop + index_d # merge drop array
    drop = np.array(drop)
    drop = np.unique(drop)
    dataset_x = np.array(x)

    dataset_x = np.delete(dataset_x, drop, 0).astype(float)
    dataset_y = np.delete(dataset_y, drop).astype(float)

    # 6311 samples
    # model construction using LinearRegression
    # Use 6100 for training and rest for testing

    train_x, train_y = dataset_x[:6101], dataset_y[:6101]
    test_x, test_y = dataset_x[6101:], dataset_y[6101:]

    clf = LinearRegression()
    clf.fit(train_x, train_y)
    predicted = clf.predict(test_x)

    # return test and train precision

    train_accuracy = clf.score(train_x, train_y)
    test_accuracy = clf.score(test_x, test_y)

    print("Train accuracy: ", train_accuracy) # 60%
    print("Test accuracy: ", test_accuracy) # 56%
