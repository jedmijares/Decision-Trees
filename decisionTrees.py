import csv
from operator import itemgetter
from anytree import Node, RenderTree, AnyNode # pip install anytree
from statistics import mode # get most common values
from math import log2
import numpy as np
import matplotlib.pyplot as plt # python -m pip install -U matplotlib

dataPoints = []

# https://realpython.com/python-csv/
with open(r'data\synthetic-4.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # ncol = len(next(csv_reader)) # Read first line and count columns
    # csv_file.seek(0)              # go back to beginning of file
    for row in csv_reader:
        dataPoints.append((float(row[0]), float(row[1]), bool(int(row[2]))))
    # print(f'Processed {line_count} lines.')

# print(dataPoints)

def ID3(examples, targetAttribute, availableAttributes):
    thisNode = AnyNode()
    thisNode.values = [float('-inf'), float('inf')]
    thisNode.label = None

    # handle base cases

    # https://thispointer.com/python-check-if-all-elements-in-a-list-are-same-or-matches-a-condition/
    allMatch = False
    if len(examples) > 0:
        allMatch = all(elem[targetAttribute] == examples[0][targetAttribute] for elem in examples)
    if allMatch:
        thisNode.label = examples[0][targetAttribute]
        return thisNode
    if not availableAttributes: # if there are no more valid attributes
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
    splitValues = [elem[bestAttribute] for elem in examples]
    splitPoint = (max(splitValues)+min(splitValues))/2
    lowerVals = []
    higherVals = []
    for point in examples:
        if point[bestAttribute] < splitPoint:
            lowerVals.append(point)
        else:
            higherVals.append(point)
    lowerAvailableAttributes = availableAttributes.copy()
    lowerAvailableAttributes.remove(bestAttribute)
    higherAvailableAttributes = lowerAvailableAttributes.copy()
    lowerNode = ID3(lowerVals, targetAttribute, lowerAvailableAttributes)
    higherNode = ID3(higherVals, targetAttribute, higherAvailableAttributes)
    lowerNode.values = [float('-inf'), splitPoint]
    lowerNode.parent = thisNode
    higherNode.values = [splitPoint, float('inf')]
    higherNode.parent = thisNode
    return thisNode

def predict(rootNode, point):
    for child in rootNode.children:
        if( child.values[0] <= point[rootNode.feature] <= child.values[1]):
            if(child.label != None):
                return child.label
            return predict(child, point)

root = ID3(dataPoints, 2, [0, 1])
# print(RenderTree(root)) 
# print(predict(root, (1,0)))
# print(RenderTree(root)) 
# print(root.children)

x = [elem[0] for elem in dataPoints]
x_min = min(x)
x_max = max(x)
y = [elem[1] for elem in dataPoints]
y_min = min(y)
y_max = max(y)
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plot_step = 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = []
for pair in np.c_[xx.ravel(), yy.ravel()]:
    Z.append(predict(root, (pair[0], pair[1])))
Z = np.asarray(Z)
Z = Z.reshape(xx.shape)

# Z = np.array()
# for i in range(len(xx)):
#     Z.insert(predict(root, (xx[i], yy[i])))
# Z = np.asarray(Z)



# Z = np.array([])
# for index1 in range(xx.shape[0]):
#     for index2 in range(xx.shape[1]):
#         Z[index1][index2] = predict(root, (xx[index1][index2], yy[index1][index2]))

    # Z = predict(root, (xx , yy[index]))
# Z = predict(root, np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

# print(xx.shape)
# print(yy.shape)

# # N = 50
# x = [elem[0] for elem in dataPoints]
# y = [elem[1] for elem in dataPoints]
# colors = [elem[2] for elem in dataPoints]

# plt.scatter(x, y, c = colors)
plt.show()