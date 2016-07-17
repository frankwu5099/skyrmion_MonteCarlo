

import sys

Tfp = open(sys.argv[1], 'r')
Hfp = open(sys.argv[2], 'r')

Tlines = Tfp.readlines()
Hlines = Hfp.readlines()

print len(Tlines), len(Hlines)
for j in Hlines:
  H = float(j)
  for i in Tlines:
    print float(i), H


print -1.0,-1.0
