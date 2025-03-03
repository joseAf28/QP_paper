import numpy as np
import scipy
import cmath
import math
import mpmath
import sys
import os

script_dir = os.path.dirname(__file__)

settings_path = os.path.join(script_dir, '..', 'settings')
settings_path = os.path.abspath(settings_path)


calculation_path = os.path.join(script_dir, '..', 'calculation')
calculation_path = os.path.abspath(calculation_path)

sys.path.append(settings_path)
sys.path.append(calculation_path)

import Constants as const
import MeanFieldv2 as MFv2

import matplotlib.pyplot as plt



def computeCurrent(lambdaValue, phiValue, voltageValue, couplingValue=const.couplingConstant, Tvalue=const.Tvalue):
    
    def currentEpsilon(epsilon, DOSfunc):
        lambdaEigen = epsilon-2.0*lambdaValue*phiValue-1j*couplingValue
        lambdaEigenConj = epsilon-2.0*lambdaValue*phiValue+1j*couplingValue
        
        beta = 1.0/Tvalue
        voltage2 = voltageValue/2.0
        
        currentMW = couplingValue**2*(MFv2.Impy(lambdaEigen, lambdaEigenConj, beta, -voltage2) - MFv2.Impy(lambdaEigen, lambdaEigenConj, beta, voltage2))
        ##! the result is a real number with zero imaginary part
        return np.real(currentMW)*DOSfunc(epsilon)
    
    currentMW = 0.0
    if const.DOSfunc == 'levels':
        energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
        for energy in energySpace:
            currentMW += currentEpsilon(energy, MFv2.DOS4func)
        currentMW = currentMW/energySpace.shape[0]
    else:
        if const.DOSfunc == 'uniform':
            integrateEpsilon = scipy.integrate.quad(currentEpsilon, -const.epsilonDOS, const.epsilonDOS, args=(MFv2.DOS1func,))[0] # -1.0, 1.0
        elif const.DOSfunc == 'lorentzian':
            integrateEpsilon = scipy.integrate.quad(currentEpsilon, -np.inf, np.inf, args=(MFv2.DOS2func,))[0]
        elif const.DOSfunc == 'gaussianMod':
            integrateEpsilon = scipy.integrate.quad(currentEpsilon, -np.inf, np.inf, args=(MFv2.DOS3func,))[0]
        
        currentMW = integrateEpsilon
        
    return currentMW



def computeCurrentEpsilon(epsilon, lambdaValue, phiValue, voltageValue, couplingValue=const.couplingConstant, Tvalue=const.Tvalue):
    lambdaEigen = epsilon-2.0*lambdaValue*phiValue-1j*couplingValue
    lambdaEigenConj = epsilon-2.0*lambdaValue*phiValue+1j*couplingValue
    
    beta = 1.0/Tvalue
    voltage2 = voltageValue/2.0
    
    currentMW = couplingValue**2*(MFv2.Impy(lambdaEigen, lambdaEigenConj, beta, -voltage2) - MFv2.Impy(lambdaEigen, lambdaEigenConj, beta, voltage2))
    ##! the result is a real number with zero imaginary part
    return np.real(currentMW)



def computeConductivity(lambdaValue, phiValue, voltageValue, couplingValue=const.couplingConstant, Tvalue=const.Tvalue):
    
    beta = 1.0/Tvalue
    
    def dImpyAux(voltage2, eigenEpsilon, signal):
        arg = (np.pi - 1j*beta*(voltage2 + eigenEpsilon)*signal)/(2.0*np.pi)
        return mpmath.polygamma(1, arg)
    
    def derivativeEpsilon(epsilon, DOSfunc):
        voltage2 = voltageValue/2.0
        
        eigenEpsilon12 = -1j*couplingValue + epsilon - 2.0*lambdaValue*phiValue
        eigenEpsilon34 = 1j*couplingValue + epsilon - 2.0*lambdaValue*phiValue
        
        value12 = dImpyAux(-voltage2, eigenEpsilon12, -1.0) + dImpyAux(voltage2, eigenEpsilon12,-1.0)
        value34 = dImpyAux(-voltage2, eigenEpsilon34, 1.0) + dImpyAux(voltage2, eigenEpsilon34, 1.0)
        
        value = DOSfunc(epsilon)*(value12 + value34)*beta*couplingValue/(8.0*np.pi**2)
        return np.real(value)
    
    dcurrentMW = 0.0
    if const.DOSfunc == 'levels':
        energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
        for nu in range(2):
            for energy in energySpace:
                dcurrentMW += derivativeEpsilon(energy, MFv2.DOS4func)
        
        dcurrentMW = np.real(dcurrentMW)/energySpace.shape[0]
    else:
        if const.DOSfunc == 'uniform':
            integrateEpsilon = scipy.integrate.quad(derivativeEpsilon, -const.epsilonDOS, const.epsilonDOS, args=(MFv2.DOS1func,))[0] # -1.0, 1.0
        elif const.DOSfunc == 'lorentzian':
            integrateEpsilon = scipy.integrate.quad(derivativeEpsilon, -np.inf, np.inf, args=(MFv2.DOS2func,))[0]
        elif const.DOSfunc == 'gaussianMod':
            integrateEpsilon = scipy.integrate.quad(derivativeEpsilon, -np.inf, np.inf, args=(MFv2.DOS3func,))[0]
        
        dcurrentMW = integrateEpsilon
        
    return dcurrentMW