import csv
from operator import itemgetter
# from anytree import Node, RenderTree, AnyNode # pip install anytree
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

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def ID3(examples, targetAttribute, availableAttributes, binCount = 8, currentDepth = 0):
    # thisNode = AnyNode()
    # thisNode.values = [float('-inf'), float('inf')]
    # thisNode.label = None
    thisNode = MyNode()

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

    # calculate entropy and find best attribute to split on
    entropies = []
    for attribute in availableAttributes:
        entropies.insert(attribute, 0)
    for label in [True, False]:
        classLabels = [elem[targetAttribute] for elem in examples]
        proportion = classLabels.count(label)/len(examples)
        if proportion != 0:
            entropies[attribute] += (-proportion * log2(proportion))
    bestAttribute = entropies.index(max(entropies))
    thisNode.feature = bestAttribute

    # place examples in bins and recurse
    examples.sort(key = lambda x: x[bestAttribute])
    minimum = float('-inf')
    for bin in list(split(examples, binCount)):
        newAvailableAttributes = availableAttributes.copy()
        newAvailableAttributes.remove(bestAttribute)
        newNode = ID3(bin, targetAttribute, newAvailableAttributes, binCount, currentDepth + 1)
        # newNode.parent = thisNode
        thisNode.children.append(newNode)
        newNode.values = [minimum, max(bin, key = lambda x: x[bestAttribute])[bestAttribute]]
        minimum = newNode.values[1]
        if max(bin, key = lambda x: x[bestAttribute])[bestAttribute] == max(examples, key = lambda x: x[bestAttribute])[bestAttribute]:
            newNode.values[1] = float('inf')
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

def readData(filename):
    dataPoints = []
    # https://realpython.com/python-csv/
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ncol = len(next(csv_reader)) # Read first line and count columns
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
    print(dataPoints)
    return dataPoints

# def readPokemon(stats, legendary):
#     statData = pd.read_csv(stats)
#     legendaryLabel = pd.read_csv(legendary)
#     # dataPoints = []
#     # https://realpython.com/python-csv/
#     # with open(filename) as csv_file:
#     #     csv_reader = csv.reader(csv_file, delimiter=',')
#     #     features = next(csv_reader) 
#     #     # csv_file.seek(0)              # go back to beginning of file
#     #     for row in csv_reader:
#     #         dataPoints.append((int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])))
#     return dataPoints, features

# dataPoints, features = readPokemon(r'data\pokemonStats.csv', r'data\pokemonLegendary.csv')
# print(dataPoints)

# dataPoints = readData(r'data/synthetic-1.csv')
dataPoints = readData(r'data/pokemonAppended.csv')
root = ID3(dataPoints, len(dataPoints[0]) - 1, [0, 1, 2, 3, 4, 5], 4)
# plot(dataPoints)

# # print(RenderTree(root))
# print(root.values)
# print(root.feature)
# print(root.label)
# # print("---------")
# for child in root.children:
#     print("-", child.values)
#     print("-", child.feature)
#     print("-", child.label)
#     # print("---------")
#     for kiddo in child.children:
#         print("--", kiddo.values)
#         print("--", kiddo.feature)
#         print("--", kiddo.label)

# check accuracy
correct = 0
total = 0
for point in dataPoints:
    if predict(root, point) == point[len(dataPoints[0]) - 1]:
        correct += 1
    total += 1
print(correct, "/", total, "=", float(correct)/total*100, "% accuracy")