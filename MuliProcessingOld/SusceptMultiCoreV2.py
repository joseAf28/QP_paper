import calculation.SusceptibilitySimpv2 as SusceptibilitySimpv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import concurrent.futures
import settings.Constants as const
import numpy as np
import scipy
import h5py
import os

##! check initial parameteres
def checkInit():
    print()
    print("QP_Project: Susceptibility Computation")
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
    else:
        pass


##! Main function
def mainSusceptibility(domain, domainRoot, domainOmega, index):
    
    pBarCopy = None
    if index == 0:
        pBarCopy = tqdm(total=2*domain.shape[0]*domainOmega.shape[0])
    
    
    resultSusceptDRmat = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')
    resultSusceptPiRmat = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')
    resultSusceptDKmat = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')
    resultSusceptPiKmat = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')

    for i in range(domain.shape[0]):
        vecR = SusceptibilitySimpv2.computeDR(domainOmega, domain[i, 0], domainRoot[i], domain[i, 1], domain[i, 2], pbar=pBarCopy)
        vecK = SusceptibilitySimpv2.computeDK(domainOmega, domain[i, 0], domainRoot[i], domain[i, 1], domain[i, 2], pbar=pBarCopy)
    
        resultSusceptPiRmat[i, :] = vecR[0]
        resultSusceptDRmat[i, :] = vecR[1]
        resultSusceptPiKmat[i, :] = vecK[0]
        resultSusceptDKmat[i, :] = vecK[1]
        
    return [resultSusceptPiRmat, resultSusceptPiKmat, resultSusceptDRmat, resultSusceptDKmat]


###! For Now, I am going to compute only the PiR and DR
if __name__ == '__main__':
    
    ##! Initial Setup
    checkInit()
    os.makedirs("outputs/" + const.foldernameOutput, exist_ok=True)
    
    ##? only for the case of the lambda root finding
    if const.modeFindRoot == 'phi':
        print("Error: must use the lambda root finding")
        quit()
    
    #! check if the Mean Field results already exist
    fileh5Name = "outputs/" + const.foldernameOutput + "/data.hdf5"

    if os.path.exists(fileh5Name):
        pass
    else:
        print("Error: file does not exist")
        print("Please run the PhiMultiCoreV2.py file first")
        quit()
    
    fileSuscept = h5py.File(fileh5Name, "a")
    groupSuscept = fileSuscept.require_group("Susceptibility")
    groupSuscept.attrs['description'] = "Charge Susceptibility Calculation"
    groupSuscept.attrs['modeFindRoot'] = const.modeFindRoot
    PiRName = "PiR"
    PiKName = "PiK"
    DRName = "DR"
    DKName = "DK"
    domainName = "Domain"

    if PiRName in groupSuscept:
        del groupSuscept[PiRName]
    if PiKName in groupSuscept:
        del groupSuscept[PiKName]
    if DRName in groupSuscept:
        del groupSuscept[DRName]
    if DKName in groupSuscept:
        del groupSuscept[DKName]
    if domainName in groupSuscept:
        del groupSuscept[domainName]
    if "Omega" in groupSuscept:
        del groupSuscept["Omega"]
    if "Voltage" in groupSuscept:
        del groupSuscept["Voltage"]
    
    ##! Create the domain
    omegaSpace = const.omegaSpaceRPA
    
    domain = fileSuscept["phiMean/Domain"][:, :]
    rootsData = fileSuscept["phiMean/Roots"][:]
    domainOmega = np.array([const.omegaSpaceRPA for i in range(const.numberCores)])
    
    ##! new domain for the susceptibility calculation
    domainVZero = domain[domain[:, 1] == 0.0]
    rootsVZero = rootsData[domain[:, 1] == 0.0]
    
    lambdaCritical = np.max(domainVZero[np.where(np.logical_and(rootsVZero > 1e-9, rootsVZero < const.max2RootRPA)),0])
    
    domainCritical = domain[domain[:, 0] == lambdaCritical]
    rootsCritical = rootsData[domain[:, 0] == lambdaCritical]
    
    lambdaUpperCritical = np.max(domainVZero[np.where(np.logical_and(domainVZero[:, 0] > lambdaCritical, domainVZero[:,0] < lambdaCritical + const.shitAboveCritical)), 0])
    domainUpperCritical = domain[domain[:, 0] == lambdaUpperCritical]
    rootsUpperCritical = rootsData[domain[:, 0] == lambdaUpperCritical]
    
    lambdaLowerCritical = np.min(domainVZero[np.where(np.logical_and(domainVZero[:, 0] < lambdaCritical, domainVZero[:,0] > lambdaCritical - const.shiftBelowCritical)), 0])
    domainLowerCritical = domain[domain[:, 0] == lambdaLowerCritical]
    rootsLowerCritical = rootsData[domain[:, 0] == lambdaLowerCritical]
    
    print("lambdaCritical: ", lambdaCritical)
    print("lambdaUpperCritical: ", lambdaUpperCritical)
    print("lambdaLowerCritical: ", lambdaLowerCritical)
    
    newDomain = np.concatenate((domainCritical, domainUpperCritical, domainLowerCritical), axis=0)
    newRoots = np.concatenate((rootsCritical, rootsUpperCritical, rootsLowerCritical), axis=0)
    
    ##! domain for each core
    domainEachCore = np.array_split(newDomain, const.numberCores)
    domainOmegaEachCore = np.array([const.omegaSpaceRPA for i in range(const.numberCores)])
    domainRootsEachCore = np.array_split(newRoots, const.numberCores)
    indexWorkers = np.arange(const.numberCores)
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor()
    resultsIter = executor.map(mainSusceptibility, domainEachCore, domainRootsEachCore, domainOmegaEachCore, indexWorkers)
    
    ##! write the results in the hdf5 file
    dataSetDR = groupSuscept.create_dataset(DRName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    dataSetPiR = groupSuscept.create_dataset(PiRName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    dataSetDK = groupSuscept.create_dataset(DKName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    dataSetPiK = groupSuscept.create_dataset(PiKName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    
    dataSetDomain = groupSuscept.create_dataset(domainName, (newDomain.shape[0], 4), dtype='float64')
    dataSetDomain[:, :3] = newDomain[:,:]
    dataSetDomain[:, 3] = newRoots
    
    groupSuscept.create_dataset("Omega", data=const.omegaSpaceRPA)
    groupSuscept.create_dataset("Voltage", data=const.voltageSpaceMF)
    
    # ##! Watch out the writing of the results
    idxBegin = 0
    for result in resultsIter:
        xShape = result[0].shape[0]
        idxEnd = idxBegin + xShape
        dataSetPiR[idxBegin:idxEnd, :] = result[0]
        dataSetPiK[idxBegin:idxEnd, :] = result[1]
        dataSetDR[idxBegin:idxEnd, :] = result[2]
        dataSetDK[idxBegin:idxEnd, :] = result[3]
        idxBegin = idxEnd
    
    fileSuscept.close()