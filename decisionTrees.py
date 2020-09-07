import csv

dataPoints = []

# https://realpython.com/python-csv/
with open(r'data\synthetic-1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print(f'\t{row[0]} \t {row[1]} \t {row[2]}.')
        dataPoints.append((float(row[0]), float(row[1]), bool(int(row[2]))))
        line_count += 1
    print(f'Processed {line_count} lines.')

print(dataPoints)