import math
total=math.log(2.2/0.2)
l=200
#a=2*total/3/(l)
a=total/(l-1)
for i in range(l):
	print "%.5f"%(0.2*math.exp(a*i))#+(a/2/l)*i*i)
#a=(1.65/1.5)**(1.0/6)
#for i in range(6+1):
#	print 1.5*a**i
