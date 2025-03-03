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
    
    
    boolInit = True
    saddleFile = None
    shotFile = None
    answer = 0
    
    while boolInit:
        try:
            answer = int(input("Zero Temperature(1) or Finite Temperature(0): "))
            if answer == 1:
                saddleFile = const.saddlePointFile
                shotFile = const.shotNoiseFile
                boolInit = False
            elif answer == 0:
                saddleFile = const.saddlePointFileTvalue
                shotFile = const.shotNoiseFileTvalue
                boolInit = False
            else:
                pass
        except:
            pass
    
    return saddleFile, shotFile


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
    saddleFile, shotFile = checkInit()
    
    ##! Read data from the hdf5 file
    with h5py.File(saddleFile, 'r') as f:
        saddle_points = f['SaddlePoint/saddle_points'][:]
        couplingValue = f['SaddlePoint/coupling'][()]
        Tvalue = f['SaddlePoint/T'][()]
        voltageSpacePlot = f['SaddlePoint/voltageSpacePlot'][:]
        lambdaSpacePlot = f['SaddlePoint/lambdaSpacePlot'][:]
        
        phi_voltage = f['SaddlePoint/phi-Voltage'][:]
        
        lambdaSamples = f['SaddlePoint/lambdaSamples'][:]
        voltageSamples = f['SaddlePoint/voltageSamples'][:]
        
        colorsVoltage = f['SaddlePoint/colorsVoltage'][:]
        colorsLambda = f['SaddlePoint/colorsLambda'][:]
    
    f.close()
    
    
    phiSamples = saddle_points[:,1]
    
    lambdaUpper = lambdaSamples[-1]
    lambdaLower = lambdaSamples[0]

    voltageZeroPoint = voltageSamples[0]
    voltageLowerPoint = voltageSamples[1]
    voltageCriticalPoint = voltageSamples[2]
    voltageUpperPoint = voltageSamples[-1]

    phiZero = phiSamples[0]
    phiLower = phiSamples[1]
    phiCritical = phiSamples[2]
    phiUpper = phiSamples[-1]

    phi_voltage_func = scipy.interpolate.interp1d(voltageSpacePlot, phi_voltage[1, :], kind='linear')
    
    ####! compute the bosonic Green Functions
    
    ####* Parameters
    # nb_cores = 9
    
    # omegaSpaceLeft =np.linspace(-6.0, -2.5, 5)
    # omegaSpaceRight = np.linspace(2.5, 6.0, 5)
    
    # omegaSpaceCenterL = np.linspace(-2.5, -1.1e-2, 20)
    # omegaSpaceCenterC = np.linspace(-1e-2, 1e-2, 6)
    # omegaSpaceCenterR = np.linspace(1.1e-2, 2.5, 20)
    # omegaSpaceCenter = np.unique(np.concatenate([omegaSpaceCenterL, omegaSpaceCenterC, omegaSpaceCenterR]))
    
    # omegaSpace = np.unique(np.concatenate([omegaSpaceLeft, omegaSpaceCenter, omegaSpaceRight]))
    
    # epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, 9)
    # # epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, 100)
    
    
    nb_cores = const.nb_cores
    omegaSpace = const.omegaSpace
    epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS_shotNoise)
    
    voltageSpace = np.array([voltageZeroPoint, voltageLowerPoint, voltageCriticalPoint, voltageUpperPoint])
    phiSpace = np.array([phiZero, phiLower, phiCritical, phiUpper])
    
    ####! Bosonic Green Function Computation - Zero Temperature for now
    
    PiRData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    PiKData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    DRData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DKData = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    pbar = tqdm(total=voltageSpace.shape[0], desc='Bosonic Green Functions')
    
    for i in range(voltageSpace.shape[0]):
        vecPiR = Susceptv2.computePiR(omegaSpace, lambdaUpper, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        vecPiK = Susceptv2.computePiK(omegaSpace, lambdaUpper, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        
        vecDR = np.reciprocal(-2.0*lambdaUpper + vecPiR)
        vecDK = Susceptv2.computeDKfunc(lambdaUpper, vecPiK, vecPiR)
        
        DRData[i,:] = vecDR
        DKData[i,:] = vecDK
        PiRData[i,:] = vecDR
        PiKData[i,:] = vecDK
        
        pbar.update(1)
        
    pbar.close()
    
    
    ###! Collect Data for each core
    domain = np.array([[lambdaUpper, voltageSpace[i], couplingValue, phiSpace[i], Tvalue] for i in range(voltageSpace.shape[0])])
    domainEachCore = np.array([domain for i in range(nb_cores)]) 
    
    epsilonSpaceEachCoreNonInt = np.array([epsilonSpace for i in range(nb_cores)])
    size_vec = omegaSpace.shape[0]
    loadSize = math.floor(size_vec/nb_cores)
    
    domainOmegaEachCoreNonInt = []
    for i in range(nb_cores):
        if i == nb_cores - 1:
            domainOmegaEachCoreNonInt.append(omegaSpace[i*loadSize:])
        else:
            domainOmegaEachCoreNonInt.append(omegaSpace[i*loadSize:(i+1)*loadSize])
    
    
    domainOmegaEachCoreInt = np.array([omegaSpace for i in range(nb_cores)])
    DRdataEachCore = np.array([DRData for i in range(nb_cores)])
    DKdataEachCore = np.array([DKData for i in range(nb_cores)])
    
    epsilonSpaceEachCoreInt = np.array_split(epsilonSpace, nb_cores)
    indexWorkers = np.arange(nb_cores)
    
    
    ###* domain for the Non-Interacting model: lambda = 0
    
    ###??? Change This Tomorrow
    domainLambdaZero = np.array([[0.0, voltageSpace[i], couplingValue, 0.0, Tvalue] for i in range(voltageSpace.shape[0])])
    domainLambdaZeroEachCore = np.array([domainLambdaZero for i in range(nb_cores)])
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_cores)
    resultsIterNonInt = executor.map(mainShotCalcOmegaNonInt, domainEachCore, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)
    resultsIterRPA = executor.map(mainShotCalcOmegaRPA, domainEachCore, domainOmegaEachCoreInt, epsilonSpaceEachCoreInt, DRdataEachCore, DKdataEachCore, indexWorkers)

    resultsIterNonIntZero = executor.map(mainShotCalcOmegaNonInt, domainLambdaZeroEachCore, domainOmegaEachCoreNonInt, epsilonSpaceEachCoreNonInt, indexWorkers)

    ##! Collect the results
    shotRetardedNonIntOmega = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonIntOmega = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedIntOmega = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshIntOmega = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedLambdaZero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshLambdaZero = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    
    idxBegin = 0
    for result in resultsIterNonInt:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedNonIntOmega[:, idxBegin:idxEnd] = result[0]
        shotKeldyshNonIntOmega[:, idxBegin:idxEnd] = result[1]
        
        idxBegin = idxEnd
    
    
    for result in resultsIterRPA:
        shotRetardedIntOmega[:, :] += result[0]
        shotKeldyshIntOmega[:, :] += result[1]
    
    
    shotRetardedIntOmega /= len(epsilonSpace)
    shotKeldyshIntOmega /= len(epsilonSpace)
    
    idxBegin = 0
    for result in resultsIterNonIntZero:
        yShape = result[0].shape[1]
        idxEnd = idxBegin + yShape
        
        shotRetardedLambdaZero[:, idxBegin:idxEnd] = result[0]
        shotKeldyshLambdaZero[:, idxBegin:idxEnd] = result[1]
        
        idxBegin = idxEnd
    
    
    ##! Write to hdf5 file
    fileh5 = h5py.File(shotFile, 'w')
    
    if "ShotNoise" in fileh5:
        del fileh5["ShotNoise"]
    
    group = fileh5.require_group("ShotNoise")
    
    group.create_dataset("omegaSpace", data=omegaSpace)
    group.create_dataset("DRdata", data=DRData)
    group.create_dataset("DKdata", data=DKData)
    
    group.create_dataset("NonIntOmegaSpace", data=omegaSpace)
    group.create_dataset("NonIntEpsilonSpace", data=epsilonSpace)
    
    group.create_dataset("NonIntRetardedOmega", data=shotRetardedNonIntOmega)
    group.create_dataset("NonIntKeldyshOmega", data=shotKeldyshNonIntOmega)
    
    group.create_dataset("IntRetardedOmega", data=shotRetardedIntOmega)
    group.create_dataset("IntKeldyshOmega", data=shotKeldyshIntOmega)
    
    group.create_dataset("NonIntRetardedLambdaZero", data=shotRetardedLambdaZero)
    group.create_dataset("NonIntKeldyshLambdaZero", data=shotKeldyshLambdaZero)
    
    fileh5.close()
