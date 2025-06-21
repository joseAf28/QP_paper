import settings.Constants as const
from tqdm import tqdm
import concurrent.futures
import numpy as np
import scipy
import os
import h5py
import math

import calculation.SusceptibilitySimpv2 as Susceptv2
import calculation.ShotNoiseRPASimbv2 as snrpa

import settings.Constants as const



##! Initial Setup
def checkInit():
    print()
    print("Paper: Shot Noise Computation")
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


def mainShotCalcOmegaNonInt(domain, domainOmega, domainEpsilon, workerIdx):

    epsilonSpace = domainEpsilon
    lambdaVec = domain[:,0]
    voltageVec = domain[:,1]
    couplingVec = domain[:,2]
    rootsVec = domain[:,3]
    Tvalue = domain[:,4]
    
    shotRetardedNonInt = np.zeros((voltageVec.shape[0], domainOmega.shape[0]), dtype=complex)
    shotKeldyshNonInt = np.zeros((voltageVec.shape[0], domainOmega.shape[0]), dtype=complex)

    pbar = None
    if workerIdx == 0:
        pbar = tqdm(total=(domainEpsilon.shape[0]*domainOmega.shape[0]*domain.shape[0]))
    
    for j in range(domainOmega.shape[0]):
        for i in range(voltageVec.shape[0]):
            
            for epsilon in epsilonSpace:
            
                shotRetardedNonInt[i,j] += snrpa.shotNoiseRetardedNonInt(epsilon, domainOmega[j], lambdaVec[i], rootsVec[i], voltageVec[i], couplingVec[i], Tvalue=Tvalue[i])
                shotKeldyshNonInt[i,j] += snrpa.shotNoiseKeldyshNonInt(epsilon, domainOmega[j], lambdaVec[i], rootsVec[i], voltageVec[i], couplingVec[i], Tvalue=Tvalue[i])
                
                if workerIdx == 0:
                    pbar.update(1)
        
            shotKeldyshNonInt[i, j]
            shotRetardedNonInt[i, j]
    
    
    if workerIdx == 0:
        pbar.close()
    
    return [shotRetardedNonInt, shotKeldyshNonInt]



def mainShotCalcOmegaRPA(domain, domainOmega, domainEpsilon, DRdata, DKdata, workerIdx):

    epsilonSpace = domainEpsilon
    omegaSpace = domainOmega
    
    lambdaVec = domain[:,0]
    voltageVec = domain[:,1]
    couplingVec = domain[:,2]
    rootsVec = domain[:,3]
    Tvalue = domain[:,4]
    
    shotRetardedInt = np.zeros((voltageVec.shape[0], domainOmega.shape[0]), dtype=complex)
    shotKeldyshInt = np.zeros((voltageVec.shape[0], domainOmega.shape[0]), dtype=complex)

    pbar = None
    if workerIdx == 0:
        pbar = tqdm(total=domainEpsilon.shape[0]*domainOmega.shape[0]*domain.shape[0])
    
    
    for j in range(len(voltageVec)):
    
        for epsilon in epsilonSpace:
            
            epsilonAux = np.array([epsilon])
            
            SKvec, SRvec = snrpa.shotNoiseInt(DRdata[j,:], DKdata[j,:], epsilonAux, omegaSpace, lambdaVec[j], rootsVec[j],\
                                                voltageVec[j], couplingVec[j], Tvalue[j], pbar=pbar)
            
            shotRetardedInt[j,:] += SRvec
            shotKeldyshInt[j,:] += SKvec
    
    if workerIdx == 0:    
        pbar.close()
    
    return [shotRetardedInt, shotKeldyshInt]




if __name__ == '__main__':
    
    ##! Initial Setup
    checkInit()
    saddleFile = "VT-space.hdf5"
    shotFile = "CurrentNoiseVT_results.hdf5"
    
    ##! Read data from the hdf5 file
    with h5py.File(saddleFile, 'r') as f:
        domain = f['VTSpace/domain'][:]
        DRDataOne = f['VTSpace/DRDataOne'][:]
        DKDataOne = f['VTSpace/DKDataOne'][:]
    
    ####* Parameters
    omegaSpace = np.array([-1e-3, 1e-5, 1e-3])
    # epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, 5)
    epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS_shotNoise)
    
    nb_cores = 9
    domainOmegaEachCoreNonInt = np.array([omegaSpace for i in range(nb_cores)])
    domainOmegaEachCoreInt = np.array([omegaSpace for i in range(nb_cores)])
    epsilonSpaceEachCore = np.array_split(epsilonSpace, nb_cores)
    indexWorkers = np.arange(nb_cores)
    
    
    ###! Collect Data for each core
    domainEachCoreOne = np.array([domain for i in range(nb_cores)])
    DRdataEachCoreOne = np.array([DRDataOne for i in range(nb_cores)])
    DKdataEachCoreOne = np.array([DKDataOne for i in range(nb_cores)])
    
    domainZero = np.array([[0.0, vec[1], vec[2], 0.0, vec[4]] for vec in domain])
    domainEachCoreZero = np.array([domainZero for i in range(nb_cores)])
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_cores)
    
    resultsIterNonIntZero = executor.map(mainShotCalcOmegaNonInt, domainEachCoreZero, domainOmegaEachCoreNonInt, epsilonSpaceEachCore, indexWorkers)
    resultsIterNonIntOne = executor.map(mainShotCalcOmegaNonInt, domainEachCoreOne, domainOmegaEachCoreNonInt, epsilonSpaceEachCore, indexWorkers)
    resultsIterRPAOne = executor.map(mainShotCalcOmegaRPA, domainEachCoreOne, domainOmegaEachCoreInt, epsilonSpaceEachCore, DRdataEachCoreOne, DKdataEachCoreOne, indexWorkers)
    
    
    ##! Collect the results
    shotRetardedNonIntZero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntZero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedNonIntOne = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntOne = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedIntOne = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshIntOne = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    
    for result in resultsIterNonIntZero:
        shotRetardedNonIntZero[:, :] += result[0]
        shotKeldyshNonIntZero[:, :] += result[1]
    for result in resultsIterNonIntOne:
        shotRetardedNonIntOne[:, :] += result[0]
        shotKeldyshNonIntOne[:, :] += result[1]
    for result in resultsIterRPAOne:
        shotRetardedIntOne[:, :] += result[0]
        shotKeldyshIntOne[:, :] += result[1]
    
    shotRetardedIntOne /= len(epsilonSpace)
    shotKeldyshIntOne /= len(epsilonSpace)
    
    ##! Write to hdf5 file
    fileh5 = h5py.File(shotFile, 'w')
    
    if "ShotNoise" in fileh5:
        del fileh5["ShotNoise"]
    
    group = fileh5.require_group("ShotNoise")
    
    group.create_dataset("retardedNonIntZero", data=shotRetardedNonIntZero)
    group.create_dataset("keldyshNonIntZero", data=shotKeldyshNonIntZero)
    group.create_dataset("retardedNonIntOne", data=shotRetardedNonIntOne)
    group.create_dataset("keldyshNonIntOne", data=shotKeldyshNonIntOne)
    
    group.create_dataset("retardedIntOne", data=shotRetardedIntOne)
    group.create_dataset("keldyshIntOne", data=shotKeldyshIntOne)
    
    group.create_dataset("domainOne", data=domain)
    group.create_dataset("domainZero", data=domainZero)
    fileh5.close()
