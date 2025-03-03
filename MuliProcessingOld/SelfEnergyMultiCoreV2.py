import calculation.SusceptibilitySimpv2 as SusceptibilitySimpv2
import calculation.BosonicNCA as BosonicNCA
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
def mainSusceptibility(domain, domainRoot, domainOmega, DRdata, DKdata, workerIdx):

    epsilonSpace = const.epsilonDOSNCA
    
    advancedSelfEnergy = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')
    keldyshSelfEnergy = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype='complex128')

    pbar = None
    if workerIdx == 0:
        pbar = tqdm(total=2*len(const.epsilonDOSNCA)*domainOmega.shape[0]*domain.shape[0])
    
    lambdaVec = domain[:,0]
    voltageVec = domain[:,1]
    couplingVec = domain[:,2]
    rootsVec = domainRoot
    
    for i in range(domain.shape[0]):
    
        for j in range(domainOmega.shape[0]):
        
            advancedSelfEnergyAux = 0.0
            keldyshSelfEnergyAux = 0.0
            
            for k in range(len(const.epsilonDOSNCA)):
                advancedSelfEnergyAux += BosonicNCA.AdvancedSelfEnergy(epsilonSpace[k], domainOmega[j], rootsVec[i], const.Tvalue, lambdaVec[i], voltageVec[i], couplingVec[i], const.omegaSpaceRPA, DRdata[i, :], DKdata[i, :])
                if pbar is not None:
                    pbar.update(1)
                keldyshSelfEnergyAux += BosonicNCA.KeldyshSelfEnergy(epsilonSpace[k], domainOmega[j], rootsVec[i], const.Tvalue, lambdaVec[i], voltageVec[i], couplingVec[i], const.omegaSpaceRPA, DRdata[i, :], DKdata[i, :])
                if pbar is not None:
                    pbar.update(1)
                
            advancedSelfEnergy[i, j] = advancedSelfEnergyAux/len(const.epsilonDOSNCA)
            keldyshSelfEnergy[i, j] = keldyshSelfEnergyAux/len(const.epsilonDOSNCA)
            
    return [np.conj(advancedSelfEnergy), keldyshSelfEnergy]


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
    groupSuscept = fileSuscept.require_group("SelfEnergy")
    groupSuscept.attrs['description'] = "Self Energy Computation"
    DRName = "Retarded"
    DKName = "Keldysh"

    if DRName in groupSuscept:
        del groupSuscept[DRName]
    if DKName in groupSuscept:
        del groupSuscept[DKName]
    if "Domain" in groupSuscept:
        del groupSuscept["Domain"]
    if "Roots" in groupSuscept:
        del groupSuscept["Roots"]
    if "Omega" in groupSuscept:
        del groupSuscept["Omega"]
    if "Voltage" in groupSuscept:
        del groupSuscept["Voltage"]
    
    ##! Create the domain
    omegaSpace = const.omegaSpaceRPA
    
    domain = fileSuscept["phiMean/Domain"][:, :]
    rootsData = fileSuscept["phiMean/Roots"][:]
    domainOmega = np.array([const.omegaSpaceSelfEnergy for i in range(const.numberCores)])
    
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
    
    newDomain = np.concatenate((domainCritical, domainUpperCritical, domainLowerCritical), axis=0)
    newRoots = np.concatenate((rootsCritical, rootsUpperCritical, rootsLowerCritical), axis=0)
    
    DRdata = fileSuscept["Susceptibility/DR"][:,:]
    DKdata = fileSuscept["Susceptibility/DK"][:,:]
    
    ##! We only consider the upper critical line
    
    idxUpper = np.where(newDomain[:, 0] == lambdaUpperCritical)[0]
    DRdataUpper = DRdata[idxUpper, :]
    DKdataUpper = DKdata[idxUpper, :]
    
    newDomain = domainUpperCritical
    newRoots = rootsUpperCritical
    
    
    
    ##! domain for each core
    domainEachCore = np.array_split(newDomain, const.numberCores)
    domainOmegaEachCore = np.array([const.omegaSpaceSelfEnergy for i in range(const.numberCores)])
    domainRootsEachCore = np.array_split(newRoots, const.numberCores)
    
    domainDREachCore = np.array_split(DRdataUpper, const.numberCores)
    domainDKEachCore = np.array_split(DKdataUpper, const.numberCores)
    
    indexWorkers = np.arange(const.numberCores)
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor()
    resultsIter = executor.map(mainSusceptibility, domainEachCore, domainRootsEachCore, domainOmegaEachCore, domainDREachCore, domainDKEachCore, indexWorkers)
    
    ##! write the results in the hdf5 file
    dataSetR = groupSuscept.create_dataset(DRName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    dataSetK = groupSuscept.create_dataset(DKName, (newDomain.shape[0], domainOmega.shape[1]), dtype='complex128')
    dataSetNewDomain = groupSuscept.create_dataset("Domain", data=newDomain)
    dataSenNewRoots = groupSuscept.create_dataset("Roots", data=newRoots)
    
    groupSuscept.create_dataset("Omega", data=const.omegaSpaceSelfEnergy)
    groupSuscept.create_dataset("Voltage", data=const.voltageSpaceMF)
    
    # ##! Watch out the writing of the results
    idxBegin = 0
    for result in resultsIter:
        xShape = result[0].shape[0]
        idxEnd = idxBegin + xShape
        dataSetR[idxBegin:idxEnd, :] = result[0]
        dataSetK[idxBegin:idxEnd, :] = result[1]
        idxBegin = idxEnd
    
    fileSuscept.close()