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
        
            shotKeldyshNonInt[i, j]/=epsilonSpace.shape[0]
            shotRetardedNonInt[i, j]/=epsilonSpace.shape[0]
    
    if workerIdx == 0:
        pbar.close()
    
    return [shotRetardedNonInt, shotKeldyshNonInt]




if __name__ == '__main__':
    
    ##! Initial Setup
    checkInit()
    shotFile = "h5_output_data/resultsCurrentNoiseNonIntV2.h5"
    
    lambdaValue = 0.0
    # omegaSpace =  np.logspace(-4.0, 3.0, 100)
    # omegaSpace = np.logspace(-4.0, 3.0, 5)
    omegaSpace =  np.linspace(-2.0, 2.0, 190)
    
    couplingValue = 0.7
    epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS_shotNoise)
    nb_cores = 9
    
    epsilonSpaceEachCoreNonInt = np.array([epsilonSpace for i in range(nb_cores)])
    size_vec = omegaSpace.shape[0]
    loadSize = math.floor(size_vec/nb_cores)
    domainOmegaEachCoreNonInt = []
    for i in range(nb_cores):
        if i == nb_cores - 1:
            domainOmegaEachCoreNonInt.append(omegaSpace[i*loadSize:])
        else:
            domainOmegaEachCoreNonInt.append(omegaSpace[i*loadSize:(i+1)*loadSize])
    
    epsilonSpaceEachCoreInt = np.array_split(epsilonSpace, nb_cores)
    indexWorkers = np.arange(nb_cores)
    
    
    ###* domain for the Non-Interacting model: lambda = 0
    Tvalue = 1e-5
    lambdaValue = 1.0
    voltageSpace = np.array([0.0, 1.0782277033434142, 1.428227703343414, 1.778227703343414])
    domainLambdaZero = np.array([[lambdaValue, voltageSpace[i], couplingValue, 0.0, Tvalue] for i in range(voltageSpace.shape[0])])
    domainLambdaZeroEachCore = np.array([domainLambdaZero for i in range(nb_cores)])
    
    Tspace = [1e-5, 2e-1, 0.45, 0.5]
    domainLambdaFinite = np.array([[lambdaValue, 0.0, couplingValue, 0.0, Tspace[i]] for i in range(voltageSpace.shape[0])])
    domainLambdaFiniteEachCore = np.array([domainLambdaFinite for i in range(nb_cores)])
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_cores)
    resultsIterNonInt = executor.map(mainShotCalcOmegaNonInt, domainLambdaZeroEachCore, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)
    # resultsIterNonInt2 = executor.map(mainShotCalcOmegaNonInt, domainLambdaFiniteEachCore, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)
    
    ##! Collect the results
    shotRetardedNonIntOmegaTZero = np.zeros((domainLambdaZero.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntOmegaTZero = np.zeros((domainLambdaZero.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedNonIntOmegaTFinite = np.zeros((domainLambdaFinite.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntOmegaTFinite = np.zeros((domainLambdaFinite.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    idxBegin = 0
    for result in resultsIterNonInt:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedNonIntOmegaTZero[:, idxBegin:idxEnd] = result[0]
        shotKeldyshNonIntOmegaTZero[:, idxBegin:idxEnd] = result[1]
        idxBegin = idxEnd
        
    idxBegin = 0
    ####! look at the argument 
    for result in resultsIterNonInt:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedNonIntOmegaTFinite[:, idxBegin:idxEnd] = result[0]
        shotKeldyshNonIntOmegaTFinite[:, idxBegin:idxEnd] = result[1]
        idxBegin = idxEnd
    
    
    
    ##! Write to hdf5 file
    fileh5 = h5py.File(shotFile, 'w')
    
    if "ShotNoise" in fileh5:
        del fileh5["ShotNoise"]
    
    group = fileh5.require_group("ShotNoise")
    
    group.create_dataset("NonIntRetardedOmegaTZero", data=shotRetardedNonIntOmegaTZero)
    group.create_dataset("NonIntKeldyshOmegaTZero", data=shotKeldyshNonIntOmegaTZero)
    
    group.create_dataset("NonIntRetardedOmegaTFinite", data=shotRetardedNonIntOmegaTFinite)
    group.create_dataset("NonIntKeldyshOmegaTFinite", data=shotKeldyshNonIntOmegaTFinite)

    fileh5.close()
