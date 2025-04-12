import numpy as np

def onehotencode(column):
    # make an array to store the unique labels and an array to store the original array as indices of the unique labels array
    unique_labels, reverse_indices = np.unique(column, return_inverse=True)
    
    # create a zero array with the same number of rows as the column and the 
    onehot = np.zeros((len(column), len(unique_labels)))
    
    # corresponding to the indices of each label, set 1
    onehot[np.arange(len(column)), reverse_indices] = 1
    
    return onehot, unique_labels

# define a function to perform z-score normalization on a particular column
def zscorenormalization(column):
    mean = np.mean(column)
    sd = np.std(column, ddof=1)
    normalized_column = (column - mean) / sd
    return normalized_column, mean, sd

    # applying z-score normalization xnorm = (x-mean)/(standard deviation) for all x
    # initializing and populating the array for normalized column

# define a function to use zscorenormalization of a column and apply it to an entire dataset
def normalizingdataset(arr):
    normalized_data = np.zeros_like(arr)
    means = []
    sds = []
    for i in range(arr.shape[1]):
        normalized_column, mean, sd = zscorenormalization(arr[:, i].astype(float))
        normalized_data[:, i] = normalized_column
        means.append(mean)
        sds.append(sd)
    return normalized_data, means, sds

# function to normalize test samples
def normalizetestsamples(test_samples, means, sds):
    normalized_test_samples = np.zeros_like(test_samples)
    for i in range(test_samples.shape[1]):
        normalized_test_samples[:, i] = (test_samples[:, i] - means[i]) / sds[i]
    return normalized_test_samples

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

dataarray = np.array(data)

# extract the last column and perform one hot encoding
fruitcolumn = dataarray[:, -1]
onehotencodedfruits, uniquefruits = onehotencode(fruitcolumn)

# remove the last column and make all the features of float type
dataarray = np.array([row[:-1] for row in data], dtype=float)

# store the normalized training set and the means and the standard deviations
normalizedfeatures, featuremeans, featuresds = normalizingdataset(dataarray)

# append the one hot encoded fruit column
dataarray = np.hstack((normalizedfeatures, onehotencodedfruits))

# make the feature matrix(3 features) and the target matrix(3 columns because of one hot encoding)
featurematrix = dataarray[:, :3].astype(float)
targetmatrix = dataarray[:, 3:]


# define a function to compute minkowski distance  between two points of n dimensions
def minkowskidist(arr1, arr2, p):
    sum = 0
    for i in range(len(arr1)):
        sq = (arr1[i]-arr2[i])**p
        sum+= sq
    return (sum)**(1/p)



# defining a class knn with hyper parameter k
class knn:
    def __init__(self, k=3, p = None):
        self.k = k
        self.p = p

    # store training data and labels
    def fit(self, xtrain, ytrain):
        self.rows = [np.array(row) for row in xtrain]
        self.labels = np.array(ytrain)

    # function to predict for one test point
    def predictone(self, x):
        # initializing and populating the list of distances
        distlist = []
        for row in self.rows:
            distlist.append(minkowskidist(x,row, self.p))

        # making a sorted list of distances with their labels as tuples
        labeldistpairs = list(zip(distlist,self.labels))
        sortedpairs = sorted(labeldistpairs, key=lambda pair: pair[0])

        # getting the labels of the smallest k distances
        smallestklabels = [pair[1] for pair in sortedpairs[:(self.k)]]

        # getting the most common label out of these smallest k labels and returning it
        labelcounts = {}
        for label in smallestklabels:
            if label in labelcounts:
                labelcounts[label] += 1
            else:
                labelcounts[label] = 1

        most_common_label = None
        max_count = 0

        for label, count in labelcounts.items():
            if count > max_count:
                most_common_label = label
                max_count = count
        return most_common_label
    

    # using the predictone function, predict for the entire testing set
    def predict(self, xtest):
        predictions = []
        for row in xtest:
            predictions.append(self.predictone(row))
        return predictions

# define a function to calculate accuracy (number of correct predicted/total number of predicted)
def accuracy(true,predicted):
    true = [str(label) for label in true]
    predicted = [str(label) for label in predicted]
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == true[i]:
            correct+=1
    acc = correct/len(predicted) 
    return acc*100

# implement the classifier on test samples
knn_classifier = knn(k=3, p=4)
knn_classifier.fit(featurematrix.tolist(), fruitcolumn.tolist())


test_samples = np.array([ [118, 6.2, 0], [160, 7.3, 1], [185, 7.7, 2]])
testsamplesnormalized = normalizetestsamples(test_samples.astype(float), featuremeans, featuresds)
ytrue = ['Banana', 'Apple', 'Orange']

predictions = knn_classifier.predict(testsamplesnormalized.tolist())
print("\nPredictions:", [str(label) for label in predictions])
print("\nAccuracy = ", accuracy(ytrue,predictions), "%\n")