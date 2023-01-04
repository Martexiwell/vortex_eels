def read_matlab_data( filename):
    csvpath =  filename + ".csv"
    f = open(csvpath, "r")
    txt = f.read()
    f.close()
    txt = txt.replace("i","j")
    
    f = open(filename+"_j.csv", "w")
    f.write(txt)
    
    data = np.loadtxt(filename+"_j.csv", dtype=complex, delimiter=",")

    return data