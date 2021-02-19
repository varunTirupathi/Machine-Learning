
############## Applying LDA, QDA and Naive Bayes classifiers for MNSIT dataset ###############


import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit as kf
from warnings import filterwarnings
# ignores LDA and QDA collinearity warns
filterwarnings("ignore", ".*collinear")

myFile = loadmat(r"E:\Data Mining and ML\Assignment 2\NumberRecognitionBigger.mat");

# Extracting the X and Y measurements from the dataset
X = myFile['X']
Y = myFile['y']

transposed_X = X.transpose()
transposed_y = Y.transpose()

reshaped_X = transposed_X.reshape(30000, 28 * 28)
reshaped_Y = transposed_y.reshape(30000)

# Boolean condition check, to get the corresponding labels of 8's and 9's
# returns true if 8's and 9's exists in columns
trainingdata_8 = (reshaped_Y == 8)
trainingdata_9 = (reshaped_Y == 9)

# 8's and 9's images corresponding to labels will be extracted
X_8 = reshaped_X[trainingdata_8, :]
X_9 = reshaped_X[trainingdata_9, :]

training_data = np.concatenate([X_8, X_9])

# preparing the labels to train the model
traininglabel_8 = np.ones(len(X_8)) * 8
traininglabel_9 = np.ones(len(X_9)) * 9

training_labels = np.concatenate([traininglabel_8, traininglabel_9])

# get_score function will fit the data to the model and will returns the accuacy score
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# getError_rate function returns error rate
def getError_rate(accuracy):
    return 1 - accuracy

kf = StratifiedKFold(n_splits=5, random_state=15, shuffle=True)

# arrays to store the error rates from all the classifiers except KNN because KNN error rates are printed directly in for loop
error_qda = []
error_lda = []
error_bayes = []
error_knn_1 = []
error_knn_5 = []
error_knn_10 = []

for train_index, test_index in kf.split(training_data, training_labels):
    X_train, X_test, y_train, y_test = training_data[train_index], training_data[test_index], training_labels[
        train_index], training_labels[test_index]

    # Linear Discriminant analysis
    accuracy_lda = (get_score(LDA(), X_train, X_test, y_train, y_test))
    error_lda.append(getError_rate(accuracy_lda))
    # Quadratic discriminant analysis
    accuracy_qda = (get_score(QDA(), X_train, X_test, y_train, y_test))
    error_qda.append(getError_rate(accuracy_qda))
    # Naive Bayes
    accuracy_bayes = (get_score(NB(), X_train, X_test, y_train, y_test))
    error_bayes.append(getError_rate(accuracy_bayes))
    # KNN classifier
    k_values = [1, 5, 10]
    for i in k_values:
        accuracy_knn = (get_score(KNN(n_neighbors=i, n_jobs=-1), X_train, X_test, y_train, y_test))
        if (i == 1):
            error_knn_1.append(getError_rate(accuracy_knn))
        elif (i == 5):
            error_knn_5.append(getError_rate(accuracy_knn))
        else:
            error_knn_10.append(getError_rate(accuracy_knn))

print('Error rates for K=1:', np.mean(error_knn_1))
print('Error rates for K=5:', np.mean(error_knn_5))
print('Error rates for K=10:', np.mean(error_knn_10))
print('Error rates of QDA:', np.mean(error_qda))
print('Error rates of LDA:', np.mean(error_lda))
print('Error rates of BAYES:', np.mean(error_bayes))


############## Applying LDA, QDA and Naive Bayes classifiers for Employee attrition dataset  ###############


import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv(r"E:\Data Mining and ML\Datasets\HR_Employee_Attrition_Data.csv")

encode = LabelEncoder()

# Encoding the categorical variables to numerical variabes
dataset['Attrition'] = encode.fit_transform(dataset['Attrition'])
dataset['Gender'] = encode.fit_transform(dataset['Gender'])
dataset['OverTime'] = encode.fit_transform(dataset['OverTime'])

# Extracting the numerical measurements from the dataset
numerical_measurements = dataset.select_dtypes(["number"])

# dropping some measurements to remove the noisy data whose cohens d measurements are infinite.
dataset_final = numerical_measurements.drop(["EmployeeCount", "StandardHours"], axis=1)

# x contains all the independent measurements
x = numerical_measurements.drop(['Attrition'], axis=1)

# y contains only predicted measurement
y = dataset['Attrition']

# get_score function will fit the data to the model and will returns the accuracy score
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# getError_rate function returns error rate
def getError_rate(accuracy):
    return 1 - accuracy

kf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)

# empty arrays to store the error rates from all the classifiers
error_qda = []
error_lda = []
error_bayes = []
error_knn_1 = []
error_knn_5 = []
error_knn_10 = []

for train_index, test_index in kf.split(x, y):
    # code adapted from https://stackoverflow.com/a/51854653/9469186
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Linear discriminant analysis
    accuracy_lda = (get_score(LDA(), X_train, X_test, y_train, y_test))
    error_lda.append(getError_rate(accuracy_lda))
    # Quadratic discriminant analysis
    accuracy_qda = (get_score(QDA(), X_train, X_test, y_train, y_test))
    error_qda.append(getError_rate(accuracy_qda))
    # Naive bayes
    accuracy_bayes = (get_score(NB(), X_train, X_test, y_train, y_test))
    error_bayes.append(getError_rate(accuracy_bayes))
    # KNN classifier
    k_values = [1, 5, 10]
    for i in k_values:
        accuracy_knn = (get_score(KNN(n_neighbors=i, n_jobs=-1), X_train, X_test, y_train, y_test))
        if (i == 1):
            error_knn_1.append(getError_rate(accuracy_knn))
        elif (i == 5):
            error_knn_5.append(getError_rate(accuracy_knn))
        else:
            error_knn_10.append(getError_rate(accuracy_knn))

print('Error rates for K=1:', np.mean(error_knn_1))
print('Error rates for K=5:', np.mean(error_knn_5))
print('Error rates for K=10:', np.mean(error_knn_10))
print('Error rates of QDA:', np.mean(error_qda))
print('Error rates of LDA:', np.mean(error_lda))
print('Error rates of BAYES:', np.mean(error_bayes))