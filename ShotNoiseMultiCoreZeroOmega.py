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
                shotFile = const.shotNoiseZeroFile
                boolInit = False
            elif answer == 0:
                saddleFile = const.saddlePointFileTvalue
                shotFile = const.shotNoiseFileZeroTvalue
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
    nb_cores = const.nb_cores
    
    omegaSpace = const.omegaSpaceZero
    epsilonSpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS_shotNoise)
    voltageSpace = const.voltageSpaceZero
    
    
    phiSpace = phi_voltage_func(voltageSpace)
    
    PiRDataZero = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    PiKDataZero = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    DRDataZero = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    DKDataZero = np.zeros((voltageSpace.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    pbar = tqdm(total=voltageSpace.shape[0], desc='Bosonic Green Functions Zero')
    
    for i in range(voltageSpace.shape[0]):
        vecPiR = Susceptv2.computePiR(omegaSpace, lambdaUpper, phi_voltage_func(voltageSpace[i]), voltageSpace[i], couplingValue, Tvalue=Tvalue)
        # vecPiK = Susceptv2.computePiK(omegaSpace, lambdaUpper, phi_voltage_func(voltageSpace[i]), voltageSpace[i], couplingValue, Tvalue=Tvalue)
        
        if Tvalue < const.Tvalue_ref:
            vecPiK = Susceptv2.computePiK(omegaSpace, lambdaUpper, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        else:
            vecPiK = Susceptv2.computePiKnum(omegaSpace, lambdaUpper, phiSpace[i], voltageSpace[i], couplingValue, Tvalue=Tvalue)
        
        
        
        vecDR = np.reciprocal(-2.0*lambdaUpper + vecPiR)
        vecDK = Susceptv2.computeDKfunc(lambdaUpper, vecPiK, vecPiR)
        DRDataZero[i,:] = vecDR
        DKDataZero[i,:] = vecDK
        PiRDataZero[i,:] = vecDR
        PiKDataZero[i,:] = vecDK
        
        pbar.update(1)
    
    pbar.close()
    
    
    ###! Collect Data for each core
    domain = np.array([[lambdaUpper, voltageSpace[i], couplingValue, phiSpace[i], Tvalue] for i in range(voltageSpace.shape[0])])
    domainEachCore = np.array([domain for i in range(nb_cores)])
    
    domainOmegaEachCoreNonInt = np.array([omegaSpace for i in range(nb_cores)])
    domainOmegaEachCoreInt = np.array([omegaSpace for i in range(nb_cores)])
    
    epsilonSpaceEachCore = np.array_split(epsilonSpace, nb_cores)
    
    DRdataEachCore = np.array([DRDataZero for i in range(nb_cores)])
    DKdataEachCore = np.array([DKDataZero for i in range(nb_cores)])
    
    indexWorkers = np.arange(nb_cores)
    
    ##! MultiProcessing
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=nb_cores)
    
    resultsIterNonInt = executor.map(mainShotCalcOmegaNonInt, domainEachCore, domainOmegaEachCoreNonInt, epsilonSpaceEachCore, indexWorkers)
    resultsIterRPA = executor.map(mainShotCalcOmegaRPA, domainEachCore, domainOmegaEachCoreInt, epsilonSpaceEachCore, DRdataEachCore, DKdataEachCore, indexWorkers)


    ##! Collect the results
    shotRetardedNonInt = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshNonInt = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    shotRetardedInt = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    shotKeldyshInt = np.zeros((domain.shape[0], omegaSpace.shape[0]), dtype=complex)
    
    
    for result in resultsIterNonInt:
        shotRetardedNonInt[:, :] += result[0]
        shotKeldyshNonInt[:, :] += result[1]
    
    for result in resultsIterRPA:
        shotRetardedInt[:, :] += result[0]
        shotKeldyshInt[:, :] += result[1]

    shotRetardedNonInt /= len(epsilonSpace)
    shotKeldyshNonInt /= len(epsilonSpace)

    shotRetardedInt /= len(epsilonSpace)
    shotKeldyshInt /= len(epsilonSpace)
    
    
    ##! Write to hdf5 file
    
    print("Tvalue: ", Tvalue)
    
    fileh5 = h5py.File(shotFile, 'w')
    
    if "ShotNoise" in fileh5:
        del fileh5["ShotNoise"]
    
    group = fileh5.require_group("ShotNoise")
    
    group.create_dataset("DRZero", data=DRDataZero)
    group.create_dataset("DKZero", data=DKDataZero)
    
    group.create_dataset("omegaInt", data=omegaSpace)
    group.create_dataset("omegaNonInt", data=omegaSpace)
    group.create_dataset("voltageSpace", data=voltageSpace)
    
    
    group.create_dataset("retardedNonInt", data=shotRetardedNonInt)
    group.create_dataset("keldyshNonInt", data=shotKeldyshNonInt)
    
    group.create_dataset("retardedInt", data=shotRetardedInt)
    group.create_dataset("keldyshInt", data=shotKeldyshInt)
    
    
    fileh5.close()
