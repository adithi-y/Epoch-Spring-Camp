Data Preprocessing
1. One-Hot Encoding
To handle categorical data (e.g., fruit types), we use one-hot encoding. This process converts categorical labels into binary vectors

It identifies unique labels in the column.

Creates a binary matrix where each row corresponds to a one-hot encoded vector.

2. Min-Max Normalization
To ensure numerical features are on the same scale (between 0 and 1), we apply min-max normalization using the formula:
The function minmaxnormalization(column) normalizes a single column, while normalizingdataset(arr) normalizes all numerical columns in the dataset.

KNN Classifier
1. Manhattan Distance
The Manhattan distance is used as the distance metric in this implementation. It calculates the sum of absolute differences between corresponding elements of two vectors:

The function manhattandist(arr1, arr2) computes this distance.

2. Implementation of KNN

The knn class implements the KNN algorithm with the following methods:
Initializes the classifier with a hyperparameter k (number of nearest neighbors).
Stores training data (xtrain) and corresponding labels (ytrain).

Predicts the label for a single test sample by:
Calculating distances from the test sample to all training samples.
Sorting distances and selecting the labels of the k nearest neighbors.
Returning the most common label among these neighbors.

Predicts labels for multiple test samples by calling predictone.


3. Accuracy Calculation
The function accuracy(true, predicted) calculates the accuracy of predictions

Example Usage
Dataset
The dataset contains fruit features such as weight (grams), size (cm), and type (categorical)

Steps
Normalize numerical features using min-max normalization.
One-hot encode the categorical labels (Fruit Type).
Train a KNN classifier on training data.
Predict fruit types for test samples.

Code Execution

Output:
Predictions: ['Banana', 'Apple', 'Orange'] 

Accuracy = 100.0 %
