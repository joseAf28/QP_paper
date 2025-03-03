import calculation.MeanFieldv2 as MFv2
import settings.Constants as const
import numpy as np
import scipy
import concurrent.futures
from tqdm import tqdm
import os
import h5py

##! Init functions
def checkInit():
    print()
    print("QP_Project: Mean Field Computation")
    print()
    print("folder: " + const.foldernameOutput)
    print("number of Cores: " + str(const.numberCores))
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
    else:
        pass


##! Main functions
def minimizeMeanField(domain, index):
    
    if index == 0:
        pbar = tqdm(total=domain.shape[0])
    
    ##? Parameters to change 
    Tvalue = const.Tvalue
    funcTemp = MFv2.ImpyTLow
    
    resultRoot = np.zeros(domain.shape[0], dtype=float)
    
    if const.modeFindRoot == 'phi':
        lambdaConstantInit = const.phiLambdaConstantInit
        for i in range(domain.shape[0]):
            result = scipy.optimize.root(MFv2.computeMeanField, lambdaConstantInit, args=(domain[i, 0], domain[i, 1], Tvalue, funcTemp, domain[i, 2]), method='lm', tol=1e-6)
            if result.fun > 1e-6 and result.success == False:
                print("Warning: result.fun > 1e-6")
                print(result)
                print("***************")
            resultRoot[i] = result.x[0]
            ## update the lambdaConstantInit for the next iteration: a better guess for the next lambda value
            lambdaConstantInit = result.x[0]
            
            if index == 0:
                pbar.update(1)
            
    elif const.modeFindRoot == 'lambda':
        phiConstantInit = const.phiConstantInit
        for i in range(domain.shape[0]):
            result = scipy.optimize.root(MFv2.computeMFLambda, phiConstantInit, args=(domain[i, 0], domain[i, 1], Tvalue, funcTemp, domain[i, 2]), method='lm', tol=1e-6)
            if result.fun > 1e-6 and result.success == False:
                print("Warning: result.fun > 1e-6")
                print(result)
                print("***************")
            resultRoot[i] = result.x[0]
            
            if index == 0:
                pbar.update(1)
    else:
        print("Error: modeFindRoot not defined")
        return quit()
    
    return resultRoot


##! Main
if __name__ == "__main__":
    
    ##! Initial Settings
    checkInit()
    os.makedirs("outputs/" + const.foldernameOutput, exist_ok=True)
    
    ##! Create the domain
    couplingSpace = const.couplingSpaceMF
    voltageSpace = const.voltageSpaceMF
    phiSpace = const.phiSpaceMF
    lambdaSpace = const.lambdaSpaceMF
    
    if const.modeFindRoot == 'phi':
        rootSpace = phiSpace
    elif const.modeFindRoot == 'lambda':
        rootSpace = lambdaSpace
    
    ###? index order - x:phi/lambda, y:voltage, z:coupling
    vecX = rootSpace
    vecY = voltageSpace
    vecZ = couplingSpace
    
    X, Y, Z = np.meshgrid(vecX, vecY, vecZ, indexing='ij')
    domain = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
    
    ##! domain for each core
    domainEachCore = np.array_split(domain, const.numberCores)
    indexWorkers = np.arange(const.numberCores)
    
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor()
    resultsIter = executor.map(minimizeMeanField, domainEachCore, indexWorkers) 
    
    
    ##! create hdf5 file
    fileh5Name = "outputs/" + const.foldernameOutput + "/data.hdf5"
    fileh5 = h5py.File(fileh5Name, 'w')
    
    group = fileh5.create_group("phiMean")
    group.attrs['numberCores'] = const.numberCores
    group.attrs['modeFindRoot'] = const.modeFindRoot
    
    dataSetRoots = group.create_dataset("Roots", (domain.shape[0]), dtype='f')
    dataSetDomain = group.create_dataset("Domain", data=domain)
    
    group.create_dataset("Voltage", data=voltageSpace)
    
    idxBegin = 0
    for result in resultsIter:
        xShape = result.shape[0]
        idxEnd = idxBegin + xShape
        dataSetRoots[idxBegin:idxEnd] = result
        idxBegin = idxEnd
        
    fileh5.close()