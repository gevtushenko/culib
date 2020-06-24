import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) != 2:
    print("Usage: {} [path to results]".format(sys.argv[0]))

with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    data = [int(row[0]) for row in reader]

bins = range(255)
plt.bar(bins, data, width=1.0)
plt.show()
