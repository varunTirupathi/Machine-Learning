############## Number recognition Mnisit dataset by using K-fold randomization. comparing the results of KNN with CNN###############

import pandas as pd
from scipy.io import loadmat
from google.colab import files
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit as kf
from tensorflow.keras.layers import Dense, Dropout,Activation,Flatten, Conv2D, MaxPooling2D

file = files.upload()

myFile =loadmat(r"NumberRecognitionBigger.mat");

X = myFile['X']
Y = myFile['y']

transposed_X = X.transpose()
transposed_y = Y.transpose()

training_data = transposed_X.reshape(30000, 28,28,1)
training_labels = transposed_y.reshape(30000,1)
image_input_shape = (28,28,1)

# get_score function will fit the data to the model and will returns the accuracy score
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# getError_rate function returns error rate
def getError_rate(accuracy):
    return 1 - accuracy

kf = StratifiedKFold(n_splits=5, random_state=15, shuffle=True)

# Empty arrays to store the KNN error values
error_knn_1 = []
error_knn_5 = []
error_knn_10 = []

for train_index, test_index in kf.split(training_data, training_labels):
    X_train, X_test, y_train, y_test = training_data[train_index], training_data[test_index], training_labels[
        train_index], training_labels[test_index]

    # KNN classifier
    k_values = [1, 5, 10]
    for i in k_values:
        accuracy_knn = (get_score(KNN(n_neighbors=i, n_jobs=-1), X_train.reshape(len(X_train), 784),
                                  X_test.reshape(len(X_test), 784), y_train.reshape(len(y_train)),
                                  y_test.reshape(len(y_test))))
        if (i == 1):
            error_knn_1.append(getError_rate(accuracy_knn))
        elif (i == 5):
            error_knn_5.append(getError_rate(accuracy_knn))
        else:
            error_knn_10.append(getError_rate(accuracy_knn))

    # CNN classifier
    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=image_input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(x=X_train, y=y_train, epochs=10)

    x = model.evaluate(X_test, y_test)
print('Error rates for K=1:', np.mean(error_knn_1))
print('Error rates for K=5:', np.mean(error_knn_5))
print('Error rates for K=10:', np.mean(error_knn_10))
print("loss", x[0])
print("accuracy", x[1])



############## Computing the cohen's d statistic which will give the impact of different measurements on the outcomes ###############

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings
# ignores divide by zero,"double_scalars" warns
filterwarnings("ignore", ".*double_scalars")

dataset = pd.read_csv(r"E:\Data Mining and ML\Datasets\HR_Employee_Attrition_Data.csv")

encode = LabelEncoder()

#Encoding the categorical variables to numerical variabes
dataset['Attrition'] = encode.fit_transform(dataset['Attrition'])
dataset['Gender'] = encode.fit_transform(dataset['Gender'])
dataset['OverTime'] = encode.fit_transform(dataset['OverTime'])

# Extracting the numerical measurements from the dataset
numerical_measurements = dataset.select_dtypes(["number"])
# dropping some measurements to remove the noisy data whose cohens d measurements are infinite.
dataset_final = numerical_measurements.drop(["EmployeeCount", "StandardHours"], axis=1)

# empty dict to store the cohens'd values
cohensd_values = {}

for column_name, column_value in dataset_final.iteritems():
    sample_group_of_interest = dataset_final.loc[dataset_final["Attrition"] == 1, column_name].values
    sample_group_not_of_interest = dataset_final.loc[dataset_final["Attrition"] == 0, column_name].values
    mean_sample_of_interest = sample_group_of_interest.mean()
    mean_sample_group_not_of_interest = sample_group_not_of_interest.mean()
    n1 = sample_group_of_interest.size
    n2 = sample_group_not_of_interest.size
    sd1 = sample_group_of_interest.std()
    sd2 = sample_group_not_of_interest.std()
    sd_average = (((n1 - 1) * (sd1 * sd1)) + (n2 - 1) * (sd2 * sd2)) / (n1 + n2 - 2)
    cohensd = abs(mean_sample_group_not_of_interest - mean_sample_of_interest) / sd_average
    cohensd_values[column_name] = cohensd

# deleting some measurements to remove the noisy data whose cohens d measurements are infinite.
del cohensd_values["Attrition"]

# sorting the cohensd_values in descending order
cohensd_values_sorted = sorted(cohensd_values, key=cohensd_values.get, reverse=True)
for r in cohensd_values_sorted:
    print(r, cohensd_values[r])

############## Applying ANN and KNN on Employee Attrition dataset  ###############

import pandas as pd
import numpy as np
from google.colab import files
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

myFile = files.upload()

dataset = pd.read_csv(r"HR_Employee_Attrition_Data.csv");
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

for train_index, test_index in kf.split(x, y):
    # code adapted from https://stackoverflow.com/a/51854653/9469186
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    error_knn_1 = []
    error_knn_5 = []
    error_knn_10 = []

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

    # ANN Classifier
    accuracy_ann = get_score(MLPClassifier(hidden_layer_sizes=5, activation="tanh", solver="lbfgs"),X_train, X_test, y_train, y_test)
    Error_rate = getError_rate(accuracy_ann)
print('Error rates for K=1:', np.mean(error_knn_1))
print('Error rates for K=5:', np.mean(error_knn_5))
print('Error rates for K=10:', np.mean(error_knn_10))
print("ANN Accuracy",accuracy_ann)
print("ANN Error Rate",Error_rate)


############## Developing the CNN architecture for MNIST dataset that is flexible and gave misclassification error rate of 0.9 ###############

import pandas as pd
from scipy.io import loadmat
from google.colab import files
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization

file = files.upload()

myFile =loadmat(r"NumberRecognitionBigger.mat");

X = myFile['X']
Y = myFile['y']

transposed_X = np.transpose(X,[2,0,1])
transposed_y = np.transpose(Y)

training_data = transposed_X.reshape(30000, 28,28,1)
training_labels = transposed_y.reshape(30000,1)
image_input_shape = (28,28,1)

# get_score function will fit the data to the model and will returns the accuracy score
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# getError_rate function returns error rate
def getError_rate(accuracy):
    return 1 - accuracy

kf = KFold(n_splits=5, random_state=15, shuffle=True)

error_knn_1 = []
error_knn_5 = []
error_knn_10 = []

for train_index, test_index in kf.split(training_data, training_labels):
    X_train, X_test, y_train, y_test = training_data[train_index], training_data[test_index], training_labels[train_index], training_labels[test_index]

# Code adapted from http://parneetk.github.io/blog/cnn-mnist/
    #CNN
    #First convolution layer
    model = Sequential()
    model.add(Convolution2D(32, (5, 5),input_shape=(28,28,1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # Second convolution layer
    model.add(Convolution2D(32, 3, 3,  border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Fully connected layer
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

model.fit(X_train,y_train, epochs=15)
__file__ = str(Path(".").absolute() / "python_predict.py")
with open("python_predict.py", "r") as file:
    script = file.read()
    exec(script)

print('Error rates for K=1:', np.mean(error_knn_1))
print('Error rates for K=5:', np.mean(error_knn_5))
print('Error rates for K=10:', np.mean(error_knn_10))
