import numpy as np

def onehotencode(column):
    # make an array to store the unique labels and an array to store the original array as indices of the unique labels array
    unique_labels, reverse_indices = np.unique(column, return_inverse=True)
    
    # create a zero array with the same number of rows as the column and the 
    onehot = np.zeros((len(column), len(unique_labels)))
    
    # corresponding to the indices of each label, set 1
    onehot[np.arange(len(column)), reverse_indices] = 1
    
    return onehot, unique_labels

# define a function for minmaxnormalization xnorm = x-xmin/xmax-xmin
def minmaxnormalization(column):
    max = np.max(column)
    min = np.min(column)
    normalizedcolumn = np.zeros(len(column))
    for i in range(len(column)):
        normalizedcolumn[i] = (column[i]-min)/(max-min)
    return normalizedcolumn

# define a function to normalize an entire dataset using the minmaxnormalization function
def normalizingdataset(arr):
    for i in range(arr.shape[1]-1):
        arr[:, i] = minmaxnormalization(arr[:, i].astype(float))
    return arr

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

normalizedarray = normalizingdataset(dataarray)

fruitcolumn = dataarray[:, -1]
onehotencodedfruits, uniquefruits = onehotencode(fruitcolumn)

dataarray = np.array([row[:-1] for row in data], dtype=float)
dataarray = np.hstack((dataarray, onehotencodedfruits))


featurematrix = dataarray[:, :3].astype(float)
targetmatrix = dataarray[:, 3:]


def manhattandist(arr1, arr2):
    sum = 0
    for i in range(len(arr1)):
        abs = np.abs((arr1[i]-arr2[i]))
        sum+= abs
    return sum




# defining a class knn with hyper parameter k
class knn:
    def __init__(self, k=3):
        self.k = k

    # store training data and labels
    def fit(self, xtrain, ytrain):
        self.rows = [np.array(row) for row in xtrain]
        self.labels = np.array(ytrain)

    # function to predict for one test point
    def predictone(self, x):
        # initializing and populating the list of distances
        distlist = []
        for row in self.rows:
            distlist.append(manhattandist(x,row))

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

knn_classifier = knn(k=3)
knn_classifier.fit(featurematrix.tolist(), fruitcolumn.tolist())


test_samples = [ [118, 6.2, 0], [160, 7.3, 1], [185, 7.7, 2]]
ytrue = ['Banana', 'Apple', 'Orange']
predictions = knn_classifier.predict(test_samples)
print("\nPredictions:", [str(label) for label in predictions], "\n")
print("\nAccuracy = ", accuracy(ytrue,predictions), "%\n")