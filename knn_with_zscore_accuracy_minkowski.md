Data Preprocessing

One-Hot Encoding
One-hot encoding is used to convert categorical labels (e.g., fruit types like "Apple", "Banana", "Orange") into binary vectors. Each label is represented as a vector where only one element is 1, and the rest are 0.

Z-Score Normalization
Z-score normalization standardizes numerical features (e.g., weight, size) so that they have a mean of 0 and a standard deviation of 1. This ensures that all features contribute equally to the distance calculations.

Test Sample Normalization
Test samples are normalized using the same mean and standard deviation calculated from the training data to ensure consistency during prediction.

KNN Classifier

Minkowski Distance

The Minkowski distance is used as the distance metric in this implementation. It generalizes other distance metrics:
Manhattan Distance (p=1)
Euclidean Distance (p=2)
Higher-order distances (p > 2)
The parameter p determines the type of distance metric used.

How KNN Works
For each test sample, calculate its distance from all training samples using Minkowski distance.
Sort the distances and select the k nearest neighbors.
Identify the most common label among these neighbors and assign it to the test sample.

Accuracy Calculation
The accuracy of predictions is calculated as the percentage of correct predictions out of the total number of predictions made by the classifier.

Example Usage
Dataset
The dataset contains information about fruits with three features:

Weight (grams)

Size (cm)

Color Code (integer)

Fruit Type (categorical)


Weight	Size	Color Code	Fruit Type
150	7.0	1	Apple
120	6.5	0	Banana
Test Samples
Three test samples are used for prediction:

[118, 6.2, 0] → Expected: Banana

[160, 7.3, 1] → Expected: Apple

[185, 7.7, 2] → Expected: Orange

Output
After training the KNN classifier with k=3 and p=4, we normalize the test samples using training data statistics and predict their labels.

Predictions:
['Banana', 'Apple', 'Orange']

Accuracy:
100%
