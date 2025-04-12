import numpy as np

# import data
data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

# convert into numpy array
dataarray = np.array(data)

# convert labels into integers
labels = dataarray[:, -1]
labels, integers = np.unique(labels, return_inverse = True)
dataarray[:,-1] = integers.astype(int)
# wine = 2, beer = 0, whiskey = 1

# make the features and target
features = dataarray[:, :-1].astype(float)
target = dataarray[:, -1]

# function to compute gini impurity of labels
def giniimpurity(array):
    length = len(array)
    _, counts = np.unique(array, return_counts=True)
    probs = counts/length
    ginisum = 0
    for element in probs:
        ginisum += element**2
    return 1-ginisum

# function to find the feature and the threshold with the least gini impurity
def findfeaturethreshold(array):
    # get the number of rows and columns in the array
    cols = array.shape[1]
    rows = array.shape[0]

    # initialise the weighted sum and returnable array
    weightedsumpgimporiginal = float('inf')
    featureandthresholdarray = []

    # iterate through every feature and every value 
    for feature in range(cols-1):
        thresholds = np.unique(array[:, feature].astype(float))
        for threshold in thresholds:

            # split the array based on the threshold
            leftone = array[array[:, feature].astype(float)<=threshold]
            rightone = array[array[:, feature].astype(float)>threshold]
            
            # get the number of rows in each for the weighted sum, and calculate the weighted sum of gini impurities
            weightedsumgimp = (leftone.shape[0]*giniimpurity(leftone[:, -1]) + rightone.shape[0]*giniimpurity(rightone[:,-1]))/rows
            
            # check if the weighted sum is the minimum of previous weighted sums and if it is then update the feature and threshold
            if weightedsumgimp <= weightedsumpgimporiginal: 
                weightedsumpgimporiginal = weightedsumgimp
                featureandthresholdarray = [feature, threshold]
    
    # return the final array containing feature and threshold
    return featureandthresholdarray

# now build the tree with this split
# define a class node which has for every decision node:
# a feature index based on which we will split 
# a threshold based on which we will split 
# a left child a right child 
# and for every leaf node has only a value which is the value itself if it is a pure node otherwise the most common value
class node:
    def __init__(self, featureindex = None, threshold = None, leftchild = None, rightchild = None,*, value = None):
        self.featureindex = featureindex
        self.threshold = threshold
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
# define a class called decision tree which has stopping parameters like maximum depth of tree or minimum samples a leaf node can have
class decisiontree:
    def __init__(self, maxdepth = 5, minsamples = 1):
        self.maxdepth = maxdepth
        self.minsamples = minsamples
        self.root = None
    
    def mostcommonvalue(self, ytrain):
        countdict = {}
        for value in ytrain:
            if value in countdict:
                countdict[value] +=1
            else:
                countdict[value] = 1
        mostcommon = None
        maxcount = 0
        for key,count in countdict.items():
            if count>maxcount:
                mostcommon = int(key)
                maxcount = count
        return mostcommon

# define a method called fit
    def fit(self, xtrain, ytrain):
        self.root = self.growtree(xtrain, ytrain)

# define a recursive helper function called growtree
    def growtree(self, xtrain, ytrain, depth = 0):
        if depth == self.maxdepth or giniimpurity(ytrain) == 0 or xtrain.shape[0]<=self.minsamples:
            leafvalue = self.mostcommonvalue(ytrain)
            return node(value = leafvalue)
        
        combineddata = np.hstack((xtrain, ytrain.reshape(-1,1)))
        feature, threshold = findfeaturethreshold(combineddata)
        leftindices = np.where(xtrain[:, feature]<=threshold)
        rightindices = np.where(xtrain[:, feature]>threshold)
        depth+=1
        lefttree = self.growtree(xtrain[leftindices], ytrain[leftindices], depth)
        righttree = self.growtree(xtrain[rightindices], ytrain[rightindices], depth)
        return node(feature, threshold, lefttree, righttree)

    
    def predict(self, x):
        currnode = self.root
        while not currnode.is_leaf_node():
            featurevalue = x[currnode.featureindex]
            if featurevalue <= currnode.threshold:
                currnode = currnode.leftchild
            else:
                currnode = currnode.rightchild
        return currnode.value
    

treemodel = decisiontree(maxdepth=5, minsamples=1)
treemodel.fit(features, target)

test_data = np.array([[6.0, 2.1, 0], [39.0, 0.05, 1], [13.0, 1.3, 1]])

predictions = [int(treemodel.predict(sample)) for sample in test_data]

label_mapping = {0: 'Beer', 1: 'Whiskey', 2: 'Wine'}
stringpreds = [label_mapping[pred] for pred in predictions]



feature_names = ['Alcohol', 'Sugar', 'Color']


def print_tree(node, depth=0):
    if node.is_leaf_node():
        print(f"{'|   ' * depth}Leaf: {label_mapping[int(node.value)]}")
    else:
        feature_name = feature_names[node.featureindex]
        print(f"{'|   ' * depth}{feature_name} <= {node.threshold}")
        print_tree(node.leftchild, depth + 1)
        print(f"{'|   ' * depth}{feature_name} > {node.threshold}")
        print_tree(node.rightchild, depth + 1)

print("\n")
print_tree(treemodel.root)
print("\n", stringpreds, "\n")