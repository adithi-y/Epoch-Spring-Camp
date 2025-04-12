Data Preprocessing
Labels: Converted to integers (Beer=0, Whiskey=1, Wine=2).
Features: Alcohol (%), Sugar (g), Color (encoded as 0 or 1).

Tree Construction
Root Node: Starts with the entire dataset.
Recursive Splitting:
For each feature and threshold, calculate Gini impurity.
Select the split that minimizes impurity.
Repeat until stopping criteria (max depth/min samples) are met.

Prediction
Traverse the tree from root to leaf using feature thresholds.
Return the majority class at the leaf node.

Example Usage
Dataset
Alcohol	Sugar	Color	Beverage
12.0	1.5	1	Wine
5.0	2.0	0	Beer
40.0	0.0	1	Whiskey
Test Samples
[6.0, 2.1, 0] → Expected: Beer

[39.0, 0.05, 1] → Expected: Whiskey

[13.0, 1.3, 1] → Expected: Wine

Results

Decision Tree Structure
Alcohol <= 36.5  
|   Alcohol <= 11.75  
|   |   Sugar <= 1.85  
|   |   |   Leaf: Beer  
|   |   Sugar > 1.85  
|   |   |   Leaf: Wine  
|   Alcohol > 11.75  
|   |   Leaf: Wine  
Alcohol > 36.5  
|   Leaf: Whiskey

Predictions
['Beer', 'Whiskey', 'Wine']
