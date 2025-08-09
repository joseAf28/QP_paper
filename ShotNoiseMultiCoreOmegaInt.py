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
    shotFile = "h5_output_data/resultsCurrentNoiseIntV2.hdf5"
    
    
    nb_cores = 9
    omegaSpace =  np.linspace(-2.0, 2.0, 214)
    epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS_shotNoise)
    
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
    domainOmegaEachCoreInt = np.array([omegaSpace for i in range(nb_cores)])
    
    
    ####! Zero Temperature case
    Tvalue = 5e-3
    couplingValue = 0.7
    voltageSpace = np.array([0.0, 1.0782277033434142, 1.428227703343414, 1.778227703343414])
    lambdaValue = 1.0
    phiSpace = np.array([0.6240326492647392, 0.42573711514556517, 0.005572809566566326, 1.687291223119121e-09])
    
    PiRData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    PiKData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DRData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DKData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    pbar = tqdm(total=voltageSpace.shape[0], desc='Bosonic Green Functions')
    for i in range(voltageSpace.shape[0]):
        vecPiR = Susceptv2.computePiR(omegaSpace, lambdaValue, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        
        if Tvalue < const.Tvalue_ref:
            vecPiK = Susceptv2.computePiK(omegaSpace, lambdaValue, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        else:
            vecPiK = Susceptv2.computePiKnum(omegaSpace, lambdaValue, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        
        vecDR = np.reciprocal(-2.0*lambdaValue + vecPiR)
        vecDK = Susceptv2.computeDKfunc(lambdaValue, vecPiK, vecPiR)
        
        DRData[i,:] = vecDR
        DKData[i,:] = vecDK
        PiRData[i,:] = vecDR
        PiKData[i,:] = vecDK
        pbar.update(1)
    pbar.close()
    
    domain = np.array([[lambdaValue, voltageSpace[i], couplingValue, phiSpace[i], Tvalue] for i in range(voltageSpace.shape[0])])
    domainEachCoreTzero = np.array([domain for i in range(nb_cores)]) 
    
    DRdataEachCoreTzero = np.array([DRData for i in range(nb_cores)])
    DKdataEachCoreTzero = np.array([DKData for i in range(nb_cores)])
    
    
    ####! Zero Voltage case
    TSpace = np.array([1e-5, 2e-1, 0.45, 0.5])
    couplingValue = 0.7
    voltageSpace = 0.0
    lambdaValue = 1.0
    phiSpace = np.array([0.6240326491198824, 0.5606227375866969, 2.475145976993829e-08, 7.883874502977921e-10])
    
    PiRData = np.zeros((TSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    PiKData = np.zeros((TSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DRData = np.zeros((TSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DKData = np.zeros((TSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    pbar = tqdm(total=TSpace.shape[0], desc='Bosonic Green Functions')
    for i in range(TSpace.shape[0]):
        vecPiR = Susceptv2.computePiR(omegaSpace, lambdaValue, phiSpace[i], voltageSpace, couplingValue, Tvalue=TSpace[i])
        
        if Tvalue < const.Tvalue_ref:
            vecPiK = Susceptv2.computePiK(omegaSpace, lambdaValue, phiSpace[i], voltageSpace, couplingValue, Tvalue=TSpace[i])
        else:
            vecPiK = Susceptv2.computePiKnum(omegaSpace, lambdaValue, phiSpace[i], voltageSpace, couplingValue, Tvalue=TSpace[i])
        
        vecDR = np.reciprocal(-2.0*lambdaValue + vecPiR)
        vecDK = Susceptv2.computeDKfunc(lambdaValue, vecPiK, vecPiR)
        
        DRData[i,:] = vecDR
        DKData[i,:] = vecDK
        PiRData[i,:] = vecDR
        PiKData[i,:] = vecDK
        pbar.update(1)
    pbar.close()
    
    domain = np.array([[lambdaValue, voltageSpace, couplingValue, phiSpace[i], TSpace[i]] for i in range(TSpace.shape[0])])
    domainEachCoreVzero = np.array([domain for i in range(nb_cores)]) 
    DRdataEachCoreVzero = np.array([DRData for i in range(nb_cores)])
    DKdataEachCoreVzero = np.array([DKData for i in range(nb_cores)])
    
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_cores)
    resultsIterNonIntTzero = executor.map(mainShotCalcOmegaNonInt, domainEachCoreTzero, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)
    resultsIterRPATzero = executor.map(mainShotCalcOmegaRPA, domainEachCoreTzero, domainOmegaEachCoreInt, epsilonSpaceEachCoreInt, DRdataEachCoreTzero, DKdataEachCoreTzero, indexWorkers)
    
    resultsIterNonIntVzero = executor.map(mainShotCalcOmegaNonInt, domainEachCoreVzero, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)
    resultsIterRPAVzero = executor.map(mainShotCalcOmegaRPA, domainEachCoreVzero, domainOmegaEachCoreInt, epsilonSpaceEachCoreInt, DRdataEachCoreVzero, DKdataEachCoreVzero, indexWorkers)


    ##! Collect the results
    shotRetardedNonIntTzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntTzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotRetardedIntTzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshIntTzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedNonIntVzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntVzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotRetardedIntVzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshIntVzero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    idxBegin = 0
    for result in resultsIterNonIntTzero:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedNonIntTzero[:, idxBegin:idxEnd] = result[0]
        shotKeldyshNonIntTzero[:, idxBegin:idxEnd] = result[1]
        idxBegin = idxEnd
    idxBegin = 0
    for result in resultsIterNonIntVzero:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedNonIntVzero[:, idxBegin:idxEnd] = result[0]
        shotKeldyshNonIntVzero[:, idxBegin:idxEnd] = result[1]
        idxBegin = idxEnd
    
    for result in resultsIterRPATzero:
        shotRetardedIntTzero[:, :] += result[0]
        shotKeldyshIntTzero[:, :] += result[1]
    for result in resultsIterRPAVzero:
        shotRetardedIntVzero[:, :] += result[0]
        shotKeldyshIntVzero[:, :] += result[1]
    
    
    shotRetardedIntTzero /= len(epsilonSpace)
    shotKeldyshIntTzero /= len(epsilonSpace)
    shotRetardedIntVzero /= len(epsilonSpace)
    shotKeldyshIntVzero /= len(epsilonSpace)
    
    
    ##! Write to hdf5 file
    fileh5 = h5py.File(shotFile, 'w')
    
    if "ShotNoise" in fileh5:
        del fileh5["ShotNoise"]
    
    group = fileh5.require_group("ShotNoise")
    
    group.create_dataset("NonIntRetardedTzero", data=shotRetardedNonIntTzero)
    group.create_dataset("NonIntKeldyshTzero", data=shotKeldyshNonIntTzero)
    group.create_dataset("NonIntRetardedVzero", data=shotRetardedNonIntVzero)
    group.create_dataset("NonIntKeldyshVzero", data=shotKeldyshNonIntVzero)
    
    group.create_dataset("IntRetardedTzero", data=shotRetardedIntTzero)
    group.create_dataset("IntKeldyshTzero", data=shotKeldyshIntTzero)
    group.create_dataset("IntRetardedVzero", data=shotRetardedIntVzero)
    group.create_dataset("IntKeldyshVzero", data=shotKeldyshIntVzero)
    
    fileh5.close()
