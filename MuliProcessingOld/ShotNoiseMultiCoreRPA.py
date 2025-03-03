import settings.Constants as const
from tqdm import tqdm
import concurrent.futures
import numpy as np
import scipy
import os
import h5py
import math

import calculation.ShotNoiseRPASimb as shotNoiseSimb


##! Initial Setup
def checkInit():
    print()
    print("QP_Project: Shot Noise Computation")
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


def mainShotCalcOmegaNonInt(domain, domainOmega, workerIdx):

    lambdaVec = domain[:,0]
    voltageVec = domain[:,1]
    couplingVec = domain[:,2]
    rootsVec = domain[:,3]

    epsilonSpace = const.epsilonDOSshotNoise

    shotNoiseNonIntOmega = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype=float)
    shotNoiseNonIntZero = np.zeros((domain.shape[0]), dtype=float)
    
    pbar = None
    if workerIdx == 0:
        pbar = tqdm(total=len(const.epsilonDOSshotNoise)*domainOmega.shape[0]*domain.shape[0])
    
    
    for i in range(domain.shape[0]):
        
        for j in range(domainOmega.shape[0]):
            shotNoiseNonIntAux = 0.0
            for k in range(len(epsilonSpace)):
                
                shotNoiseNonIntAux += shotNoiseSimb.shotNoiseNonInteracting(domainOmega[j], epsilonSpace[k], domain[i,0], domain[i,3], domain[i,1], domain[i,2])
                
                if pbar is not None:
                    pbar.update(1)
            
            shotNoiseNonIntOmega[i, j] = shotNoiseNonIntAux/len(epsilonSpace)
        
        shotNoiseNonIntAux = 0.0
        for k in range(len(epsilonSpace)):
            shotNoiseNonIntAux += shotNoiseSimb.shotNoiseNonInteracting(0.0, epsilonSpace[k], domain[i,0], domain[i,3], domain[i,1], domain[i,2])
            
        shotNoiseNonIntZero[i] = shotNoiseNonIntAux/len(epsilonSpace)
        
    return [shotNoiseNonIntOmega, shotNoiseNonIntZero]


def mainShotCalcOmegaRPA(domain, domainOmega, DRdata, DKdata, workerIdx):

    lambdaVec = domain[:,0]
    voltageVec = domain[:,1]
    couplingVec = domain[:,2]
    rootsVec = domain[:,3]

    epsilonSpace = const.epsilonDOSshotNoise

    shotNoiseRPAOmega = np.zeros((domain.shape[0], domainOmega.shape[0]), dtype=float)
    shotNoiseRPAZero = np.zeros((domain.shape[0]), dtype=float)
    
    pbar = None
    if workerIdx == 0:
        pbar = tqdm(total=len(const.epsilonDOSshotNoise)*domainOmega.shape[0]*domain.shape[0])
    
    
    for i in range(domain.shape[0]):
        
        for j in range(domainOmega.shape[0]):
            shotNoiseRPAAux = 0.0
            for k in range(len(epsilonSpace)):
                    
                shotNoiseRPAAux += shotNoiseSimb.shotNoiseRPA(domainOmega[j], epsilonSpace[k], domain[i,0], domain[i,3], domain[i,1], domain[i,2], \
                    np.conj(DRdata[i]), DRdata[i], DKdata[i])
                
                if pbar is not None:
                    pbar.update(1)
            
            ##! the complex part is of the order 10^-9
            shotNoiseRPAOmega[i, j] = np.real(shotNoiseRPAAux)/len(epsilonSpace)
        
    
        shotNoiseRPAAux = 0.0
        for k in range(len(epsilonSpace)):
            shotNoiseRPAAux += shotNoiseSimb.shotNoiseRPA(0.0, epsilonSpace[k], domain[i,0], domain[i,3], domain[i,1], domain[i,2], \
                    np.conj(DRdata[i]), DRdata[i], DKdata[i])
            
        shotNoiseRPAZero[i] = np.real(shotNoiseRPAAux)/len(epsilonSpace)
        
    return [shotNoiseRPAOmega, shotNoiseRPAZero]



if __name__ == '__main__':
    
    ##! It works with lambda Mode only
    
    ##! Initial Setup
    checkInit()
    os.makedirs("outputs/" + const.foldernameOutput, exist_ok=True)

    ##! Read data from hdf5 file and create new datasets
    fileh5Name = "outputs/" + const.foldernameOutput + "/data.hdf5"
    fileh5 = h5py.File(fileh5Name, 'a')
    
    shotNoiseGroup = fileh5.require_group("ShotNoise")

    shotNoiseRPAStr = 'RPA'
    shotNoiseNonIntStr = 'NonInt'
    
    if shotNoiseRPAStr in shotNoiseGroup:
        del shotNoiseGroup[shotNoiseRPAStr]
    if shotNoiseNonIntStr in shotNoiseGroup:
        del shotNoiseGroup[shotNoiseNonIntStr]
    
    
    ##! Read data from the hdf5 file
    newDomain = fileh5["Susceptibility/Domain"][:,:]
    
    DR = fileh5["Susceptibility/DR"][:, :]
    DK = fileh5["Susceptibility/DK"][:, :]
    
    ###? S(omega, V) Results
    ##! domain for each core
    domainEachCore = np.array_split(newDomain, const.numberCores)
    
    domainOmegaEachCoreRPA = np.array([const.omegaSpaceShotNoiseRPA for i in range(const.numberCores)])
    domainOmegaEachCoreNonInt = np.array([const.omegaSpaceShotNoiseNonInt for i in range(const.numberCores)])
    
    domainDReachCore = np.array_split(DR, const.numberCores)
    domainDKeachCore = np.array_split(DK, const.numberCores)
    indexWorkers = np.arange(const.numberCores)
    
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor()
    
    resultsIterNonInt = executor.map(mainShotCalcOmegaNonInt, domainEachCore, domainOmegaEachCoreNonInt, indexWorkers)
    resultsIterRPA = executor.map(mainShotCalcOmegaRPA, domainEachCore, domainOmegaEachCoreRPA, domainDReachCore, domainDKeachCore, indexWorkers)
    
    
    ##! Collect the results
    shotNoiseNonIntOmega = np.zeros((newDomain.shape[0], const.omegaSpaceShotNoiseNonInt.shape[0]), dtype=float)
    shotNoiseRPAOmega = np.zeros((newDomain.shape[0], const.omegaSpaceShotNoiseRPA.shape[0]), dtype=float)
    
    shotNoiseNonIntZero = np.zeros((newDomain.shape[0]), dtype=float)
    shotNoiseRPAZero = np.zeros((newDomain.shape[0]), dtype=float)
    
    
    idxBegin = 0
    for result in resultsIterNonInt:
        xShape = result[0].shape[0]
        idxEnd = idxBegin + xShape
        
        shotNoiseNonIntOmega[idxBegin:idxEnd] = result[0]
        shotNoiseNonIntZero[idxBegin:idxEnd] = result[1]
        
        idxBegin = idxEnd
        
    idxBegin = 0
    for result in resultsIterRPA:
        xShape = result[0].shape[0]
        idxEnd = idxBegin + xShape
        
        shotNoiseRPAOmega[idxBegin:idxEnd] = result[0]
        shotNoiseRPAZero[idxBegin:idxEnd] = result[1]
        
        idxBegin = idxEnd
    
    
    ##! Write to hdf5 file
    
    rpaGroup = shotNoiseGroup.create_group(shotNoiseRPAStr)
    nonIntGroup = shotNoiseGroup.create_group(shotNoiseNonIntStr)
    
    rpaGroup.create_dataset("Omega", data=shotNoiseRPAOmega)
    rpaGroup.create_dataset("Zero", data=shotNoiseRPAZero)
    
    nonIntGroup.create_dataset("Omega", data=shotNoiseNonIntOmega)
    nonIntGroup.create_dataset("Zero", data=shotNoiseNonIntZero)
    
    
    nonIntGroup.create_dataset("Voltage", data=const.voltageSpaceMF)
    nonIntGroup.create_dataset("OmegaSpaceNonInt", data=const.omegaSpaceShotNoiseNonInt)
    rpaGroup.create_dataset("Voltage", data=const.voltageSpaceMF)
    rpaGroup.create_dataset("OmegaSpaceRPA", data=const.omegaSpaceShotNoiseRPA)
    
    fileh5.close()
