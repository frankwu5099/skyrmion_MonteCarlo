"""
A tool for generating configuration file with the grid points from Tsfile and Hsfile.
arg 1:
    parameters (json)
arg 2:
    Temperatures for T axis
arg 3:
    H for H axis

"""


import sys
import json


with open(sys.argv[1],'r') as f:
    params = json.load(f)

config = {}
config["parameters"] = params


Ts = []
Hs = []
if len(sys.argv)>3:
    Tfp = open(sys.argv[2], 'r')
    Hfp = open(sys.argv[3], 'r')

    Tlines = Tfp.readlines()
    Hlines = Hfp.readlines()
    NumTaxis = len(Tlines)
    NumHaxis = len(Hlines)



    for j in Hlines:
      H = float(j)
      for i in Tlines:
          Ts.append(float(i))
          Hs.append(H)
    Tfp.close()
    Hfp.close()
else:
    THfp = open(sys.argv[2],'r')
    for i, line in enumerate(THfp.readlines()):
        if i == 0:
            NumTaxis, NumHaxis = line.strip().split(" ")
            NumTaxis = int(NumTaxis)
            NumHaxis = int(NumHaxis)
        else:
            T, H = line.strip().split(" ")
            if float(T) < 0:
                break
            Ts.append(float(T))
            Hs.append(float(H))


ensemble = {}

ensemble["NumTaxis"] = NumTaxis
ensemble["NumHaxis"] = NumHaxis
ensemble["Ts"] = Ts
ensemble["Hs"] = Hs
config['ensemble'] = ensemble


with open("config.json", 'w+') as f:
    json.dump(config, f, indent=4)

print("succeed")
