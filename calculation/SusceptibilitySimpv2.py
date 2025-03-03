import scipy
import numpy as np
import cmath
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname(__file__)

settings_path = os.path.join(script_dir, '..', 'settings')
settings_path = os.path.abspath(settings_path)

mf_path = os.path.join(script_dir, '..', 'calculation')
mf_path = os.path.abspath(mf_path)

sys.path.append(settings_path)
sys.path.append(mf_path)

import Constants as const
import MeanFieldv2 as MFv2


##! Susceptibility Calculation
def computePiR(omegaSpace, lambdaConstant, phiFieldInit, voltageValue, couplingConstant=const.couplingConstant, pbar=None, Tvalue=const.TvalueRPA):
    ##! Retarded Susceptibility - general case
    ## Parameters
    chemicalPotential = [voltageValue/2.0, -voltageValue/2.0]
    beta = [1.0/Tvalue, 1.0/Tvalue]

    DOSoperations = {'uniform': MFv2.DOS1func, 'lorentzian': MFv2.DOS2func, 'gaussianMod': MFv2.DOS3func, 'levels': MFv2.DOS4func}

    valueRetardedVec = np.zeros(omegaSpace.shape[0], dtype=complex)

    ## it also includes the sum over the left and right chemical potentials
    def unitPiR(epsilon, omega, funcZ, funcZp):
        valueInt = 0.0
        
        for nu in range(len(chemicalPotential)):
            valueLambda1 = funcZ(epsilon - 2.0*lambdaConstant*phiFieldInit - 1j*couplingConstant) + omega
            valueLambda2 = funcZp(epsilon - 2.0*lambdaConstant*phiFieldInit - 1j*couplingConstant)
            
            valueInt += MFv2.Impy(valueLambda1, valueLambda2, beta[nu], chemicalPotential[nu])
            
        return valueInt
        
    funcIden = lambda x: x
        
    PiA11 = lambda omega, epsilon: unitPiR(epsilon, -omega, funcIden, funcIden)
    PiA22 = lambda omega, epsilon: -1.0*unitPiR(epsilon, omega, np.conj, np.conj)
    PiA12 = lambda omega, epsilon: -1.0*unitPiR(epsilon, -omega, funcIden, np.conj)
    PiA21 = lambda omega, epsilon: unitPiR(epsilon,omega, np.conj, funcIden)
    PiAtot = lambda epsilon, DOSfunc, omega: DOSfunc(epsilon)*(PiA11(omega, epsilon) + PiA12(omega, epsilon) + PiA21(omega, epsilon) + PiA22(omega, epsilon))

    if const.DOSfunc == 'levels':

        energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
        for i in range(omegaSpace.shape[0]):
            
            valueAdvancedAux = 0.0
            for energy in energySpace:
                valueAdvancedAux += PiAtot(energy, MFv2.DOS4func, omegaSpace[i])
            
            valueRetardedVec[i] = valueAdvancedAux/len(energySpace)
            
            if pbar is not None:
                pbar.update(1)
    else:
        PiAtotReal = lambda epsilon, DOSfunc, omega: np.real(PiAtot(epsilon, DOSfunc, omega))
        PiAtotImag = lambda epsilon, DOSfunc, omega: np.imag(PiAtot(epsilon, DOSfunc, omega))
            
        if const.DOSfunc == 'uniform':
                integrateEpsilonReal = lambda omega: scipy.integrate.quad(PiAtotReal, -const.epsilonDOS, const.epsilonDOS, args=(MFv2.DOS1func, omega), limit=140)[0]
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiAtotImag, -const.epsilonDOS, const.epsilonDOS, args=(MFv2.DOS1func, omega))[0]
        elif const.DOSfunc == 'lorentzian':
                integrateEpsilonReal = lambda omega: scipy.integrate.quad(PiAtotReal, -np.inf, np.inf, args=(MFv2.DOS2func, omega), limit=140)[0]
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiAtotImag, -np.inf, np.inf, args=(MFv2.DOS2func, omega), limit=140)[0]
        elif const.DOSfunc == 'gaussianMod':
                integrateEpsilonReal = lambda omega: scipy.integrate.quad(PiAtotReal, -np.inf, np.inf, args=(MFv2.DOS3func, omega))[0]
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiAtotImag, -np.inf, np.inf, args=(MFv2.DOS3func, omega))[0]


        for i in range(omegaSpace.shape[0]):
            
            ##! avoid having an indetermination at omega = 0
            if omegaSpace[i] == 0.0:
                omegaSpace[i] = 1e-7
            
            valueAux = integrateEpsilonReal(omegaSpace[i]) + 1j*integrateEpsilonImag(omegaSpace[i])
            
            valueRetardedVec[i] = valueAux
            
            if pbar is not None:
                pbar.update(1)
    
    return valueRetardedVec*2.0*lambdaConstant**2*1j


####! Susceptibility Calculation
def GKGKterm(omega, epsilon, mul, mulp, lambdaValue, phiValue, couplingValue):
    Gamma2 = couplingValue**2
    Gamma = couplingValue
    
    term1 = (2 * np.pi * omega) / (Gamma * (4 * Gamma2 * omega + omega**3))
    term2 = np.sign(mul - mulp - omega) * (2 / (np.sqrt(Gamma2) * (4 * Gamma2 * omega + omega**3)))
    
    arc_term1 = omega * np.arctan((epsilon - mul - 2 * lambdaValue*phiValue) / Gamma)
    arc_term2 = -omega * np.arctan((epsilon - mulp - 2 * lambdaValue*phiValue) / Gamma)
    arc_term3 = omega * np.arctan((epsilon - mul - 2 * lambdaValue*phiValue + omega) / Gamma)
    arc_term4 = omega * np.arctan((-epsilon + mulp + 2 * lambdaValue*phiValue + omega) / Gamma)
    
    log_term1 = -Gamma * np.log(Gamma2 + (-epsilon + mul + 2 * lambdaValue*phiValue)**2)
    log_term2 = -Gamma *np.log(Gamma2 + (-epsilon + mulp + 2 * lambdaValue*phiValue)**2)
    log_term3 = Gamma *np.log(Gamma2 + (epsilon - mul - 2 * lambdaValue*phiValue + omega)**2)
    log_term4 = Gamma *np.log(Gamma2 + (-epsilon + mulp + 2 * lambdaValue*phiValue + omega)**2)
    
    result = term1 + term2 * (arc_term1 + arc_term2 + arc_term3 + arc_term4 + log_term1 + log_term2 + log_term3 + log_term4)
    
    return Gamma2*result


def GK_GK(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    voltage2 = voltageValue/2.0
    value = GKGKterm(omega, epsilon, -voltage2, -voltage2, lambdaValue, phiValue, couplingValue) + GKGKterm(omega, epsilon, voltage2, voltage2, lambdaValue, phiValue, couplingValue) +\
            GKGKterm(omega, epsilon, -voltage2, voltage2, lambdaValue, phiValue, couplingValue) + GKGKterm(omega, epsilon, voltage2, -voltage2, lambdaValue, phiValue, couplingValue)
    
    return -value/(2.0*cmath.pi)


def computePiK_aux(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    ### Keldysh Susceptibility - Zero Temperature Limit
    value1 = GK_GK(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    ### GA.GR + GR.GA term
    num2 = 8.0*cmath.pi*couplingValue
    den2 = 4.0*couplingValue**2 + omega**2
    
    value2 = num2/den2/(2.0*cmath.pi)
    
    return (value1 + value2)



def computePiK(omegaSpace, lambdaConstant, phiFieldInit, voltageValue, couplingConstant=const.couplingConstant, pbar=None, Tvalue=const.TvalueRPA):
    
    ##! Keldysh Susceptibility - Zero Temperature Limit
    ## Parameters
    chemicalPotential = [voltageValue/2.0, -voltageValue/2.0]
    beta = [1.0/Tvalue, 1.0/Tvalue]

    DOSoperations = {'uniform': MFv2.DOS1func, 'lorentzian': MFv2.DOS2func, 'gaussianMod': MFv2.DOS3func, 'levels': MFv2.DOS4func}

    valueKeldyshVec = np.zeros(omegaSpace.shape[0], dtype=complex)
    PiKtot = lambda epsilon, DOSfunc, omega: DOSfunc(epsilon)*computePiK_aux(epsilon, omega, lambdaConstant, phiFieldInit, voltageValue, couplingConstant, Tvalue)

    if const.DOSfunc == 'bilevel':

        energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
        for i in range(omegaSpace.shape[0]):
            
            valueAux = 0.0
            
            if np.abs(omegaSpace[i]) < 1e-7:
                omegaSpace[i] = 1e-5
            
            for energy in energySpace:
                valueAux += PiKtot(energy, MFv2.DOS4func, omegaSpace[i])
            
            valueKeldyshVec[i] = valueAux/len(energySpace)
            
            if pbar is not None:
                pbar.update(1)
    else:
        PiKtotImag = lambda epsilon, DOSfunc, omega: np.real(PiKtot(epsilon, DOSfunc, omega))
            
        if const.DOSfunc == 'uniform':
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiKtotImag, -const.epsilonDOS, const.epsilonDOS, args=(MFv2.DOS1func, omega))[0]
        elif const.DOSfunc == 'lorentzian':
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiKtotImag, -np.inf, np.inf, args=(MFv2.DOS2func, omega), limit=140)[0]
        elif const.DOSfunc == 'gaussianMod':
                integrateEpsilonImag = lambda omega: scipy.integrate.quad(PiKtotImag, -np.inf, np.inf, args=(MFv2.DOS3func, omega))[0]


        for i in range(omegaSpace.shape[0]):
            
            ##! avoid having an indetermination at omega = 0
            if omegaSpace[i] == 0.0:
                omegaSpace[i] = 1e-7
            
            valueAux = integrateEpsilonImag(omegaSpace[i])
            
            valueKeldyshVec[i] = valueAux
            
            if pbar is not None:
                pbar.update(1)
    
    return valueKeldyshVec*2.0*lambdaConstant**2*1j



def computePiKnum(omegaSpace, lambdaConstant, phiFieldInit, voltageValue, couplingConstant, Tvalue=const.TvalueRPA, pbar=None):
    
    ##! Keldish Susceptibility non-zero temperature case
    ## Parameters
    voltage2 = voltageValue/2.0
    beta = 1.0/Tvalue
    beta2 = beta/2.0

    DOSoperations = {'uniform': MFv2.DOS1func, 'lorentzian': MFv2.DOS2func, 'gaussianMod': MFv2.DOS3func, 'levels': MFv2.DOS4func}
    
    valueKeldishVec = np.zeros(omegaSpace.shape[0], dtype=complex)

    def GK(omega, epsilonValue):
        voltage2 = 0.5*voltageValue
        num = -1j*couplingConstant*(np.tanh(beta2 * (omega + voltage2))+ np.tanh(beta2 * (omega - voltage2)))
        den = (omega - epsilonValue + 2.0*lambdaConstant*phiFieldInit)**2 + couplingConstant**2
        
        return num/den
    
    def integrandFunc(nu, omega, epsilonValue):
        omega2 = omega/2.0
        value = (GK(nu + omega2, epsilonValue)*GK(nu - omega2, epsilonValue))/(2.0*np.pi)
        return np.real(value)
    
    def integralFunc(omega, epsilon):
        valueIntegrand = scipy.integrate.quad(integrandFunc, -np.inf, np.inf, args=(omega, epsilon), limit=100)
        
        num2 = 8.0*cmath.pi*couplingConstant
        den2 = 4.0*couplingConstant**2 + omega**2
        
        return valueIntegrand[0] + num2/den2/(2.0*cmath.pi)
    
    
    def PiKDOSfunc(DOSfunc, limInt, omega): 
        def integrandFunc(epsilon):
            return DOSfunc(epsilon)*(integralFunc(omega, epsilon))
        
        valueIntegrand = scipy.integrate.quad(integrandFunc, -limInt, limInt, limit=70)
        return (valueIntegrand[0])
    
    
    def computeKeldysh(omega):
        if const.DOSfunc == 'levels':
            
            valueKeldish = 0.0
            energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
            for i in range(energySpace.shape[0]):
                valueKeldish += integralFunc(omega, energySpace[i])
                
            valueKeldish /= energySpace.shape[0]
            
        elif const.DOSfunc == 'uniform':
            valueKeldish = PiKDOSfunc(MFv2.DOS1func, const.epsilonDOS, omega)
        elif const.DOSfunc == 'lorentzian':
            valueKeldish = PiKDOSfunc(MFv2.DOS2func, np.inf, omega)
        elif const.DOSfunc == 'gaussianMod':
            valueKeldish = PiKDOSfunc(MFv2.DOS3func, np.inf, omega)
        else:
            pass
        
        return valueKeldish
    
    for i in range(omegaSpace.shape[0]):
        
        ##! avoid having an indetermination at omega = 0
        if omegaSpace[i] == 0.0:
            omegaSpace[i] = 1e-7
        
        valueKeldishVec[i] = computeKeldysh(omegaSpace[i])
        if pbar is not None:
            pbar.update(1)
        
    return valueKeldishVec*2.0*lambdaConstant**2*1j



##! Charge Susceptibility calculation
def computeDKfunc(lambdaValue, PiKdata, PiRdata):
    
    den = (-2.0*lambdaValue + np.real(PiRdata))**2 + np.imag(PiRdata)**2
    
    if den.size > 1:
        mask = np.abs(den) < 1e-7
        den[mask] = 1e-5
    
    DKdata = -PiKdata/den
    return DKdata


def computeXK(PiKdata, PiRdata, lambdaValue):
    den = 1.0 - PiRdata/(2.0*lambdaValue)
    denReal = np.real(den)
    denImag = np.imag(den)
    return -PiKdata*np.reciprocal(denReal**2 + denImag**2)


def computeXR(PiRdata, lambdaValue):
    
    den = 1.0 - PiRdata/(2.0*lambdaValue)
    return -PiRdata*np.reciprocal(den)


def computeRatioFD(PiRdata, PiKdata, omegaSpace):
    
    omegaMin = np.abs(omegaSpace).min()
    ratioFD = np.imag(PiKdata)/np.imag(PiRdata)
    ratioFD = np.where(np.abs(omegaSpace) <= omegaMin, np.nan, ratioFD)
    return ratioFD

