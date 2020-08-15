import sys
import numpy as np




f = open(sys.argv[1],'r')
for line in f.readlines():
    if "Temperature Set" in line:
        data = line.split(':')[-1].strip()
        data = data.split()
        data = np.unique(np.array([float(i) for i in data]))
        print(data)
        for i in data:
            print(i)
