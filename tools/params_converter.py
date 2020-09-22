"""
A tool for converting previous config file for paramters to new format (paramter.json).
"""
import json
import sys

def general_convert(s:str):
    """
    Convert string s to int if possible.
    Or conver string s to float if possible.
    """
    try:
        out = int(s)
        return out
    except:
        pass
    try:
        out = float(s)
        return out
    except:
        pass
    return s
if __name__ == "__main__":
    params = {}
    with open(sys.argv[1],'r') as f:
        for line in f.readlines():
            key, value = line.strip().split(" ")
            value = general_convert(value)
            params[key] = value
    with open(sys.argv[2], 'w+') as f:
        json.dump(params, f, indent=4)







