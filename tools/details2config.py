
import sys
import json

with open(sys.argv[1],'r') as f:
    config = {}
    params = {}
    ensemble = {}
    for line in f.readlines():
        if "SpinSize" in line:
            params["Size"] = int(line.strip().split('=')[-1])
        if "Thickness" in line:
            params["Size_z"] = int(line.strip().split('=')[-1])
        if "A =" in line:
            params["A"] = float(line.strip().split('=')[-1])
        if "D_R" in line:
            params["DR"] = float(line.strip().split('=')[-1])
        if "D_D" in line:
            params["DD"] = float(line.strip().split('=')[-1])
        if "Bin Size" in line:
            params["BIN_SIZE"] = int(line.strip().split('=')[-1])
        if "Bin Number" in line:
            params["BIN_NUM"] = int(line.strip().split('=')[-1])
        if "Equilibration N =" in line:
            params["EQUI_N"] = int(line.strip().split('=')[-1])
        if "Equilibration Ni" in line:
            params["EQUI_Ni"] = int(line.strip().split('=')[-1])
        if "PT frequency" in line:
            params["PTF"] = float(line.strip().split('=')[-1])
        if "f_CORR" in line:
            params["f_CORR"] = int(line.strip().split('=')[-1])
        if "CORR_N" in line:
            params["CORR_N"] = int(line.strip().split('=')[-1])
        if 'Temperature Set' in line:
            ensemble["Ts"]=\
                    [ float(i) for i in line.strip().split(": ")[-1].split("  ")]
        if 'field Set' in line:
            ensemble["Hs"]=\
                    [ float(i) for i in line.strip().split(": ")[-1].split("  ")]
            ensemble['NumHaxis'] = len(set(ensemble["Hs"]))
        if 'Pnum' in line:
            Pnum = int(line.strip().split('=')[-1])
    params["relax_N"] = 0
    ensemble['NumTaxis'] = Pnum//ensemble['NumHaxis']
    config["parameters"] = params
    config["ensemble"] = ensemble
with open(sys.argv[2], 'w+') as f:
    json.dump(config, f, indent=4)


