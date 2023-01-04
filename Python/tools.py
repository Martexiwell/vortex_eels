import numpy as np

def read_matlab_data( filename):
    csvpath =  filename + ".csv"
    f = open(csvpath, "r")
    txt = f.read()
    f.close()
    txt = txt.replace("i","j")
    
    f = open(filename+"_j.csv", "w")
    f.write(txt)
    
    data = np.loadtxt(filename+"_j.csv", dtype=complex, delimiter=" ")

    return data

def read_data_metadata( filename):
    """"
    Reads data and metadata saved as string to header of file
    """
    csvpath =  filename
    data = np.loadtxt(csvpath, dtype = float)
    f = open(csvpath, "r")
    txt = f.readline()[2:].replace("'","\"")#.replace("datetime.datetime","")
    metadata = eval(txt)
    f.close()

    return data, metadata