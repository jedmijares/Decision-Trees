import csv
from operator import itemgetter
from anytree import Node, RenderTree, AnyNode # pip install anytree
from statistics import mode # get most common values
from math import log2
# from collections import Counter

dataPoints = []

# https://realpython.com/python-csv/
with open(r'data\synthetic-3.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # line_count = 0
    for row in csv_reader:
        # print(f'\t{row[0]} \t {row[1]} \t {row[2]}.')
        dataPoints.append((float(row[0]), float(row[1]), bool(int(row[2]))))
        # line_count += 1
    # print(f'Processed {line_count} lines.')

# print(dataPoints)

def ID3(examples, targetAttribute, availableAttributes):
    thisNode = AnyNode()

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

    entropies = []
    for attribute in availableAttributes:
        entropies.insert(attribute, 0)
    for label in [True, False]:
        classLabels = [elem[targetAttribute] for elem in examples]
        proportion = classLabels.count(label)/len(examples)
        if proportion != 0:
            # print(entropies)
            # print(availableAttributes)
            # print("--------")
            entropies[attribute] += (-proportion * log2(proportion))
    bestAttribute = entropies.index(max(entropies))
    splitValues = [elem[bestAttribute] for elem in examples]
    thisNode.splitPoint = (max(splitValues)+min(splitValues))/2
    lowerVals = []
    higherVals = []
    for point in examples:
        if point[bestAttribute] < thisNode.splitPoint:
            lowerVals.append(point)
        else:
            higherVals.append(point)
    lowerAvailableAttributes = availableAttributes.copy()
    lowerAvailableAttributes.remove(bestAttribute)
    higherAvailableAttributes = lowerAvailableAttributes.copy()
    lowerNode = ID3(lowerVals, targetAttribute, lowerAvailableAttributes)
    higherNode = ID3(higherVals, targetAttribute, higherAvailableAttributes)
    lowerNode.parent = thisNode
    higherNode.parent = thisNode
    return thisNode

    # print(entropies)
    # print(bestAttribute)

print(RenderTree(ID3(dataPoints, 2, [0, 1]))) 