#this is the program to generate the field strength as geometric series.
#You should give:
#A : the ratio of the last difference of the strength to the first difference of the strength.
#Hmax : the highest field strength (we start from the zero field.)
#N: number of the fields.
####
# it would decide the common ratio and the start term and give a geometric series as the values of the strength of the field.


import numpy as np
import sys

if (len(sys.argv) < 2):
	print "A, Hmax, N"
	quit()
# read the params
A = float(sys.argv[1])
Hmax = float(sys.argv[2])
N = int(sys.argv[3])



# determine the ratio
r = A**(1/float(N-1))
# determing the start term
a = (r-1.0)/(A-1) * Hmax
print 0.0
for i in range(N-1):
	print a*(r**(i+1) -1)/(r -1)


