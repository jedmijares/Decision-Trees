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

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def ID3(examples, targetAttribute, availableAttributes, binCount = 8):
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

    # place examples in bins and recurse
    examples.sort(key = lambda x: x[bestAttribute])
    minimum = float('-inf')
    for bin in list(split(examples, binCount)):
        newAvailableAttributes = availableAttributes.copy()
        newAvailableAttributes.remove(bestAttribute)
        newNode = ID3(bin, targetAttribute, newAvailableAttributes, binCount)
        newNode.parent = thisNode
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

root = ID3(dataPoints, 2, [0, 1])
print(RenderTree(root)) 

x = [elem[0] for elem in dataPoints]
y = [elem[1] for elem in dataPoints]
classLabels = [elem[2] for elem in dataPoints]
colors = []
for i in range(len(classLabels)):
    if classLabels[i]:
        colors.append('b')
    else:
        colors.append('r')

x = [elem[0] for elem in dataPoints]
x_min = min(x) - 1
x_max = max(x) + 1
y = [elem[1] for elem in dataPoints]
y_min = min(y) - 1
y_max = max(y) + 1
plot_step = 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = []
for pair in np.c_[xx.ravel(), yy.ravel()]:
    Z.append(predict(root, (pair[0], pair[1])))
Z = np.asarray(Z)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)

plt.scatter(x, y, c = colors)
plt.show()