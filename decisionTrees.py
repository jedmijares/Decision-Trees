import csv
from operator import itemgetter
from anytree import Node, RenderTree, AnyNode # pip install anytree

dataPoints = []

# https://realpython.com/python-csv/
with open(r'data\synthetic-1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print(f'\t{row[0]} \t {row[1]} \t {row[2]}.')
        dataPoints.append((float(row[0]), float(row[1]), bool(int(row[2]))))
        line_count += 1
    # print(f'Processed {line_count} lines.')

print(dataPoints)

def ID3(examples, targetAttribute, attributes):
    thisNode = AnyNode()
    # for point in examples:
    #     print(point[2])

    # https://thispointer.com/python-check-if-all-elements-in-a-list-are-same-or-matches-a-condition/
    allMatch = False
    if len(examples) > 0:
        allMatch = all(elem[2] == examples[0][2] for elem in examples)
    print(allMatch)


ID3(dataPoints, 'a', 3)

# # https://www.tutorialspoint.com/get-first-element-with-maximum-value-in-list-of-tuples-in-python
# range0 = max(dataPoints, key=itemgetter(0))[0] - min(dataPoints, key=itemgetter(0))[0]
# range1 = max(dataPoints, key=itemgetter(1))[1] - min(dataPoints, key=itemgetter(1))[1]

# print(range0)
# print(range1)

# discriminationRange = -1
# if range0 > range1:
#     discriminationRange = 0
# else:
#     discriminationRange = 1

# depth1 = Node((discriminationRange, range0))

# for pre, fill, node in RenderTree(depth1):
#     print("%s%s" % (pre, node.name))

# print(depth1)