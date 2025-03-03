import settings.Constants as const
import calculation.MeanFieldv2 as MFv2
import calculation.CurrentMeirv2 as currentv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import h5py
from tqdm import tqdm


## Function used to compute the current:
funcCurrent = currentv2.computeCurrent
funcDCurrent = currentv2.computeConductivity

# check initial parameteres
def checkInit():
    print()
    print("QP_Project: Meir Wingreen Current Computation")
    print()
    print("folder: " + const.foldernameOutput)
    print()
    
    boolInit = True
    while boolInit:
        try:
            answer = int(input("Do you want to continue? [y: 1/ n: 0]: "))
            if answer == 1 or answer == 0:
                boolInit = False
        except:
            pass
    
    if answer == 0:
        print("Programm stopped.")
        return quit()
    elif answer == 1:
        pass
    else:
        pass


if __name__ == '__main__':
    
    checkInit()
    os.makedirs("outputs/" + const.foldernameOutput, exist_ok=True)
    
    fileName = "outputs/" + const.foldernameOutput + "/data.hdf5"
    fileCurrent = h5py.File(fileName, "a")
    
    groupCurrent = fileCurrent.require_group("current")
    groupCurrent.attrs['description'] = "current: with the MF green fuctions"

    currentName = "MeirWingreenMF"
    dCurrentName = "dMeirWingreenMF"
    if currentName in groupCurrent:
        del groupCurrent[currentName]
    if dCurrentName in groupCurrent:
        del groupCurrent[dCurrentName]

    domain = fileCurrent["phiMean/Domain"][:, :]
    domainRoot = fileCurrent["phiMean/Roots"][:]
    
    dataSetCurrent = groupCurrent.create_dataset(currentName, (domain.shape[0]), dtype='float')
    dataSetDCurrent = groupCurrent.create_dataset(dCurrentName, (domain.shape[0]), dtype='float')
    
    pbar = tqdm(total=domain.shape[0])
    for i in range(domain.shape[0]):
        if const.modeFindRoot == 'phi':
            result = funcCurrent(domainRoot[i], domain[i, 0], domain[i, 1], domain[i, 2])
            resultD = funcDCurrent(domainRoot[i], domain[i, 0], domain[i, 1], domain[i, 2])
        elif const.modeFindRoot == 'lambda':
            result = funcCurrent(domain[i, 0], domainRoot[i], domain[i, 1], domain[i, 2])
            resultD = funcDCurrent(domain[i, 0], domainRoot[i], domain[i, 1], domain[i, 2])
            
        dataSetCurrent[i] = result
        dataSetDCurrent[i] = resultD
        pbar.update(1)

    fileCurrent.close()