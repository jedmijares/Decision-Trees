# Gerard (Jed) Mijares
# Machine Learning
# Fall 2020
# Decision Trees

import csv
from statistics import mode # get most common values
from math import log2
import numpy as np
import matplotlib.pyplot as plt # python -m pip install -U matplotlib
import random # for randomness during cross-validation

# node class for use in decision tree
class MyNode:
    def __init__(self):
        self.values = [float('-inf'), float('inf')]
        self.label = None
        self.children = []
        self.feature = None
        self.depth = None

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# implementation of ID3 algorithm
def ID3(examples, targetAttribute, availableAttributes, currentDepth, maxDepth = 3, binCount = 8, featureList = []):
    thisNode = MyNode()
    thisNode.depth = currentDepth

    # handle base cases
    # https://thispointer.com/python-check-if-all-elements-in-a-list-are-same-or-matches-a-condition/
    allMatch = False
    if len(examples) > 0:
        allMatch = all(elem[targetAttribute] == examples[0][targetAttribute] for elem in examples)
    else: # examples is empty
        return thisNode
    if allMatch:
        thisNode.label = examples[0][targetAttribute]
        return thisNode
    if (not availableAttributes) | (currentDepth >= maxDepth): # if there are no more valid attributes
        thisNode.label = mode(label[targetAttribute] for label in examples) # set label to most common value
        return thisNode

    # calculate entropy of the current set
    entropy = 0
    for label in [True, False]:
        classLabels = [elem[targetAttribute] for elem in examples]
        proportion = classLabels.count(label)/len(examples)
        if proportion != 0:
            entropy += (-proportion * log2(proportion))

    # calculate information gain of each available attribute to find the best one to split on
    infoGains = [0] * (max(availableAttributes) + 1)
    for attribute in availableAttributes:
        infoGains[attribute] = entropy
        examples.sort(key = lambda x: x[attribute])
        if (len(featureList) > 0): 
            # if a list of feature names was passed and this feature represents a Pokemon type
            if (featureList[attribute].startswith("Type")): 
                trueVals = []
                falseVals = []
                for point in examples:
                    if point[attribute] == True:
                        trueVals.append(point)
                    else:
                        falseVals.append(point)
                for bin in [trueVals, falseVals]:
                    if bin: # if bin is not empty
                        binEntropy = 0 # entropy of this bin
                        for label in [True, False]:
                            classLabels = [elem[targetAttribute] for elem in bin]
                            proportion = classLabels.count(label)/len(bin)
                            if proportion != 0:
                                binEntropy += (-proportion * log2(proportion))
                        infoGains[attribute] -= len(bin)/len(examples)*binEntropy
        else: # this attribute is float data
            for bin in list(split(examples, binCount)):
                if bin: # if bin is not empty
                    binEntropy = 0 # entropy of this bin
                    for label in [True, False]:
                        classLabels = [elem[targetAttribute] for elem in bin]
                        proportion = classLabels.count(label)/len(bin)
                        if proportion != 0:
                            binEntropy += (-proportion * log2(proportion))
                    infoGains[attribute] -= len(bin)/len(examples)*binEntropy
    bestAttribute = infoGains.index(max(infoGains))
    thisNode.feature = bestAttribute

    # place examples in bins and recurse
    examples.sort(key = lambda x: x[bestAttribute])

    if (len(featureList) > 0): 
        # if a list of feature names was passed and the best feature represents a Pokemon type
        if (featureList[bestAttribute].startswith('Type')): 
            trueVals = []
            falseVals = []
            for point in examples:
                if point[bestAttribute] == True:
                    trueVals.append(point)
                else:
                    falseVals.append(point)
            for bin in [trueVals, falseVals]:
                newAvailableAttributes = availableAttributes.copy()
                newAvailableAttributes.remove(bestAttribute)
                newNode = ID3(bin, targetAttribute, newAvailableAttributes, maxDepth=maxDepth, currentDepth=currentDepth+1, binCount=binCount, featureList=featureList)
                thisNode.children.append(newNode)
                # set the "range" of this split to two copies of either True or False
                newNode.values = [bin[0][bestAttribute], bin[0][bestAttribute]] 
            return thisNode
    # else, this is float data            
    minimum = float('-inf')
    for bin in list(split(examples, binCount)):
        if bin: # if bin is not empty
            newAvailableAttributes = availableAttributes.copy()
            newAvailableAttributes.remove(bestAttribute)
            newNode = ID3(bin, targetAttribute, newAvailableAttributes, currentDepth+1, maxDepth=maxDepth, binCount=binCount, featureList=featureList)
            thisNode.children.append(newNode)
            newNode.values = [minimum, max(bin, key = lambda x: x[bestAttribute])[bestAttribute]]
            minimum = newNode.values[1]
            if max(bin, key = lambda x: x[bestAttribute])[bestAttribute] == max(examples, key = lambda x: x[bestAttribute])[bestAttribute]:
                # if the maximum of this bin is the maximum of all the examples, this bin should cover values to infinity
                newNode.values[1] = float('inf') 
    return thisNode

# given the tree and a point, predict class label of the point
def predict(rootNode, point):
    for child in rootNode.children:
        if( child.values[0] <= point[rootNode.feature] <= child.values[1]):
            if(child.label != None):
                return child.label
            return predict(child, point)

# referred to https://scikit-learn.org/0.15/auto_examples/tree/plot_iris.html
def plot(points, rootNode, subplot = False, plotIndex = 0):
    if(subplot):
        ax = plt.subplot(2, 2, plotIndex)
    x = [elem[0] for elem in points]
    y = [elem[1] for elem in points]
    # set bounds of plat a little further than the max and min
    x_min = min(x) - 1
    x_max = max(x) + 1
    y_min = min(y) - 1
    y_max = max(y) + 1
    plot_step = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    predictions = []
    for pair in np.c_[xx.ravel(), yy.ravel()]:
        predictions.append(predict(rootNode, (pair[0], pair[1])))
    predictions = np.asarray(predictions)
    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, cmap=plt.cm.RdBu)
    classLabels = [elem[2] for elem in points]
    colors = []
    for i in range(len(classLabels)):
        if classLabels[i]:
            colors.append('b')
        else:
            colors.append('r')
    plt.scatter(x, y, c = colors)
    return ax

def readData(filename, hasHeader = False):
    dataPoints = []
    features = None
    # https://realpython.com/python-csv/
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        if(hasHeader):
            features = next(csv_reader)
        for row in csv_reader:
            newPoint = row
            for i in range(len(newPoint)):
                # store data as either a Boolean or a float
                if (newPoint[i] == '0') | (newPoint[i] == 'FALSE'):
                    newPoint[i] = False
                elif (newPoint[i] == '1') | (newPoint[i] == 'TRUE'):
                    newPoint[i] = True
                else:
                    newPoint[i] = float(newPoint[i])
            newPoint = tuple(newPoint)
            dataPoints.append(newPoint)
    if hasHeader:
        return dataPoints, features
    return dataPoints

BINCOUNT = 8

# create/plot trees for synthetic data
plotIndex = 1
for fileName in [r'data/synthetic-1.csv', r'data/synthetic-2.csv', r'data/synthetic-3.csv', r'data/synthetic-4.csv']:
    featureNames = ['x', 'y']
    dataPoints = readData(fileName)
    root = ID3(dataPoints, len(dataPoints[0]) - 1, list(range(len(dataPoints[0]) - 1)), 0, maxDepth=3, binCount=BINCOUNT, featureList=featureNames)
    subplot = plot(dataPoints, root, True, plotIndex)
    subplot.set_title(fileName)
    plotIndex += 1

    # check accuracy
    correct = 0
    total = 0
    for point in dataPoints:
        if predict(root, point) == point[len(dataPoints[0]) - 1]:
            correct += 1
        total += 1
    print(correct, "/", total, "=", float(correct)/total*100, "% accuracy")
plt.suptitle("Plots")
plt.savefig(r'media/plots.png')
plt.close()

# create/plot trees with cross-validation
random.seed(4) # set seed for repeatability
foldSize = 50 # size of a fold, 200/50 = 4 folds
plotIndex = 1
for fileName in [r'data/synthetic-1.csv', r'data/synthetic-2.csv', r'data/synthetic-3.csv', r'data/synthetic-4.csv']:
    featureNames = ['x', 'y']
    dataPoints = readData(fileName)
    random.shuffle(dataPoints) # shuffle data so true and false class labels are shuffled
    correctCounts = [0, 0, 0] # number of correctly predicted examples at each depth
    for maximumDepth in [1, 2]:
        correct = 0
        for i in range(int(len(dataPoints)/foldSize)):
            foldPoints = dataPoints.copy()
            del foldPoints[i*foldSize:(i+1)*foldSize]
            root = ID3(foldPoints, len(dataPoints[0]) - 1, list(range(len(dataPoints[0]) - 1)), 0, maxDepth=maximumDepth, binCount=BINCOUNT, featureList=featureNames)
            # check accuracy by testing the removed points
            for point in dataPoints[i*foldSize:(i+1)*foldSize]:
                if predict(root, point) == point[len(dataPoints[0]) - 1]:
                    correct += 1
        print("For depth", maximumDepth, ":", correct, "correct")
        correctCounts[maximumDepth] = correct
    # best maximum depth is the first one to have the highest accuracy overall
    bestMaxDepth = correctCounts.index(max(correctCounts)) 
    featureNames = ['x', 'y']
    dataPoints = readData(fileName)
    root = ID3(dataPoints, len(dataPoints[0]) - 1, list(range(len(dataPoints[0]) - 1)), 0, maxDepth=bestMaxDepth, binCount=BINCOUNT, featureList=featureNames)

    subplot = plot(dataPoints, root, True, plotIndex)
    subplot.set_title(fileName)
    plotIndex += 1

    # check accuracy
    correct = 0
    total = 0
    for point in dataPoints:
        if predict(root, point) == point[len(dataPoints[0]) - 1]:
            correct += 1
        total += 1
    print(correct, "/", total, "=", float(correct)/total*100, "% accuracy")
plt.suptitle('Plots With Cross-Validation')
plt.savefig(r'media/cross-validated-plots.png')

# create tree for Pokemon data
featureNames = None
dataPoints, featureNames = readData(r'data/pokemonAppended.csv', True)
root = ID3(dataPoints, len(dataPoints[0]) - 1, list(range(len(dataPoints[0]) - 1)), 0, maxDepth=3, binCount=BINCOUNT, featureList=featureNames)

# check accuracy
correct = 0
total = 0
for point in dataPoints:
    if predict(root, point) == point[len(dataPoints[0]) - 1]:
        correct += 1
    total += 1
print(correct, "/", total, "=", float(correct)/total*100, "% accuracy")
