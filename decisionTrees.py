import csv
from operator import itemgetter
from statistics import mode # get most common values
from math import log2
import numpy as np
import matplotlib.pyplot as plt # python -m pip install -U matplotlib
import pandas as pd # pip install pandas

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

def ID3(examples, targetAttribute, availableAttributes, currentDepth, binCount = 8, featureList = []):
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
    if (not availableAttributes) | (currentDepth >= 3): # if there are no more valid attributes
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
            if (featureList[attribute].startswith("Type")): # if a list of feature names was passed and this feature represents a Pokemon type
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
        if (featureList[bestAttribute].startswith('Type')): # if a list of feature names was passed and the best feature represents a Pokemon type
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
                newNode = ID3(bin, targetAttribute, newAvailableAttributes, currentDepth=currentDepth+1, binCount=binCount, featureList=featureList)
                thisNode.children.append(newNode)
                newNode.values = [bin[0][bestAttribute], bin[0][bestAttribute]] # set the "range" of this split to two copies of either True or False
            return thisNode
    # else, this is float data            
    minimum = float('-inf')
    for bin in list(split(examples, binCount)):
        if bin: # if bin is not empty
            newAvailableAttributes = availableAttributes.copy()
            newAvailableAttributes.remove(bestAttribute)
            newNode = ID3(bin, targetAttribute, newAvailableAttributes, currentDepth+1, binCount, featureList=featureList)
            thisNode.children.append(newNode)
            newNode.values = [minimum, max(bin, key = lambda x: x[bestAttribute])[bestAttribute]]
            minimum = newNode.values[1]
            if max(bin, key = lambda x: x[bestAttribute])[bestAttribute] == max(examples, key = lambda x: x[bestAttribute])[bestAttribute]:
                newNode.values[1] = float('inf') # if the maximum of this bin is the maximum of all the examples, this bin should cover values to infinity
    return thisNode

def predict(rootNode, point):
    for child in rootNode.children:
        if( child.values[0] <= point[rootNode.feature] <= child.values[1]):
            if(child.label != None):
                return child.label
            return predict(child, point)

def plot(points):
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
    Z = []
    for pair in np.c_[xx.ravel(), yy.ravel()]:
        Z.append(predict(root, (pair[0], pair[1])))
    Z = np.asarray(Z)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    classLabels = [elem[2] for elem in points]
    colors = []
    for i in range(len(classLabels)):
        if classLabels[i]:
            colors.append('b')
        else:
            colors.append('r')
    plt.scatter(x, y, c = colors)
    plt.show()

def readData(filename, hasHeader = False):
    dataPoints = []
    # https://realpython.com/python-csv/
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # ncol = len(next(csv_reader)) # Read first line and count columns
        if(hasHeader):
            features = next(csv_reader)
        # csv_file.seek(0)              # go back to beginning of file
        for row in csv_reader:
            # print(row)
            newPoint = row
            for i in range(len(newPoint)):
                if (newPoint[i] == '0') | (newPoint[i] == 'FALSE'):
                    newPoint[i] = False
                elif (newPoint[i] == '1') | (newPoint[i] == 'TRUE'):
                    newPoint[i] = True
                else:
                    newPoint[i] = float(newPoint[i])
            newPoint = tuple(newPoint)
            dataPoints.append(newPoint)
    # print(dataPoints)
    if hasHeader:
        return dataPoints, features
    return dataPoints

featureNames = ['x', 'y']
dataPoints = readData(r'data/synthetic-4.csv')
# dataPoints, featureNames = readData(r'data/pokemonAppended2.csv', True)
root = ID3(dataPoints, len(dataPoints[0]) - 1, list(range(len(dataPoints[0]) - 1)), 0, binCount=8, featureList=featureNames)

plot(dataPoints)

# print(RenderTree(root))
print(root.depth)
print(root.values)
# print(root.feature)
print(featureNames[root.feature])
print(root.label)
print("---------")
for child in root.children:
    print("-", child.depth)
    print("-", child.values)
    # print("-", child.feature)
    if child.feature != None:
        print("-", featureNames[child.feature])
    print("-", child.label)
    print("---------")
    for kiddo in child.children:
        print("--", kiddo.depth)
        print("--", kiddo.values)
        # print("--", kiddo.feature)
        if kiddo.feature != None:
            print("--", featureNames[kiddo.feature])
        print("--", kiddo.label)
        print("---------")
        for kid in kiddo.children:
            print("---", kid.depth)
            print("---", kid.values)
            # print("---", kid.feature)
            if kid.feature != None:
                print("---", featureNames[kid.feature])
            print("---", kid.label)
            print("---------")
            for baby in kid.children:
                print("----", baby.depth)
                print("----", baby.values)
                # print("---", baby.feature)
                if baby.feature != None:
                    print("----", featureNames[baby.feature])
                print("----", baby.label)
                print("---------")
                for infant in baby.children:
                    print("-----", infant.depth)
                    print("-----", infant.values)
                    # print("-----", infant.feature)
                    if infant.feature:
                        print("-----", featureNames[infant.feature])
                    print("-----", infant.label)
                    print("---------")

# check accuracy
correct = 0
total = 0
for point in dataPoints:
    if predict(root, point) == point[len(dataPoints[0]) - 1]:
        correct += 1
    total += 1
print(correct, "/", total, "=", float(correct)/total*100, "% accuracy")