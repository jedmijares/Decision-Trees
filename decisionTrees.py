import csv
from operator import itemgetter

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

# https://www.tutorialspoint.com/get-first-element-with-maximum-value-in-list-of-tuples-in-python
range0 = max(dataPoints, key=itemgetter(0))[0] - min(dataPoints, key=itemgetter(0))[0]
range1 = max(dataPoints, key=itemgetter(1))[1] - min(dataPoints, key=itemgetter(1))[1]

print(range0)
print(range1)