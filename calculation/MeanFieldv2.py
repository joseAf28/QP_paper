import numpy as np
import cmath
import math
import scipy
import matplotlib.pyplot as plt
import sys
import os

script_dir = os.path.dirname(__file__)

settings_path = os.path.join(script_dir, '..', 'settings')
settings_path = os.path.abspath(settings_path)

sys.path.append(settings_path)

import Constants as const


###?  Auxiliary functions
###! Definition of the \Psi function
def Impy(z, zp, beta, mu): 
    arrayZ = [(cmath.pi - 1j*beta*(z - mu)*np.sign(np.imag(z)))/(2*cmath.pi)]
    arrayZp = [(cmath.pi - 1j*beta*(zp - mu)*np.sign(np.imag(zp)))/(2*cmath.pi)]
    
    den =(2.0*cmath.pi*(z - zp))
    
    valueAux = (-2.0*scipy.special.psi(arrayZ) + 2.0*scipy.special.psi(arrayZp))/den
    return valueAux

###! Definition of the \Psi function in the limit of T -> 0
def ImpyTLow(z, zp, beta, mu):
    arrayZ = [(-1j*(z - mu)*np.sign(np.imag(z)))/(2*cmath.pi)]
    arrayZp = [(-1j*(zp - mu)*np.sign(np.imag(zp)))/(2*cmath.pi)]
    
    psiArrayZ = np.array([cmath.log(z) for z in arrayZ])
    psiArrayZp = np.array([cmath.log(zp) for zp in arrayZp])
    
    valueAux = (-2.0*psiArrayZ + 2.0*psiArrayZp)/(2.0*cmath.pi*(z - zp))
    return valueAux


###! Density of States functions from the Quantum Dot 
DOS1func = lambda epsilon: 0.5
DOS2func = lambda epsilon: const.LambdaDOS/(epsilon*epsilon + const.LambdaDOS*const.LambdaDOS)/(cmath.pi)

def DOS3func(epsilon):
    expSigma = const.nuDOS+1
    expAux = 0.5 + 0.5*const.nuDOS

    funcAux = math.pow(abs(epsilon), const.nuDOS)*math.exp(-epsilon*epsilon/(2.0*const.sigmaDOS))
    norm = math.pow(2.0, expAux)*const.sigmaDOS**expSigma*scipy.special.gamma(0.5*expSigma)
    return funcAux/norm

DOS4func = lambda epsilon: 1.0


###! Compute the Mean Field function - \Gamma \propto \mathbb{1}
def computeMeanField(lambdaConstant, phiFieldInit, voltageValue, Tvalue, ImpyFunc=Impy, couplingConstant=const.couplingConstant): 
    
    chemicalPotential = [voltageValue/2.0, -voltageValue/2.0]
    beta = [1.0/Tvalue, 1.0/Tvalue]

    ##! Obtain the Keldish Green Function
    def unitGK(epsilon, DOSfunc, nu):
        valuelambda = epsilon - 2.0*lambdaConstant*phiFieldInit - 1j*couplingConstant
        valueInt = ImpyFunc(valuelambda, np.conj(valuelambda), beta[nu], chemicalPotential[nu])
        return np.real(valueInt[0]*couplingConstant*DOSfunc(epsilon))

    DOSoperations = {'uniform': DOS1func, 'lorentzian': DOS2func, 'gaussianMod': DOS3func, 'levels': DOS4func}
    
    traceKeldishGF = 0.0
    if const.DOSfunc == 'levels':
        energySpace = np.linspace(-const.epsilonDOS, const.epsilonDOS, const.nbDOS)
        for nu in range(len(chemicalPotential)):
            for energy in energySpace:
                traceKeldishGF += unitGK(energy, DOS4func, nu)
                
        phiFieldKeldish = -1.0*traceKeldishGF/energySpace.shape[0]
    else:
        if const.DOSfunc == 'uniform':
            integrateEpsilon = lambda nu: scipy.integrate.quad(unitGK, -const.epsilonDOS, const.epsilonDOS, args=(DOS1func, nu))[0] # -1, §
        elif const.DOSfunc == 'lorentzian':
            integrateEpsilon = lambda nu: scipy.integrate.quad(unitGK, -np.inf, np.inf, args=(DOS2func, nu))[0]
        elif const.DOSfunc == 'gaussianMod':
            integrateEpsilon = lambda nu: scipy.integrate.quad(unitGK, -np.inf, np.inf, args=(DOS3func, nu))[0]
        else:
            print("Error: DOSfunc not defined")
            return quit()
        
        for nu in range(len(chemicalPotential)):
            traceKeldishGF += integrateEpsilon(nu)

        phiFieldKeldish = -1.0*traceKeldishGF
    
    diffPhiValue = phiFieldKeldish - phiFieldInit

    return diffPhiValue


computeMFBeta = lambda TValue, lambdaHam, phiFieldInit, voltageValue, ImpyFunc :computeMeanField(lambdaHam, phiFieldInit, voltageValue, TValue, ImpyFunc)

computeMFLambda = lambda phiFieldInit, lambdaHam, voltageValue, TValue, ImpyFunc, couplingConstant: computeMeanField(lambdaHam, phiFieldInit, voltageValue, TValue, ImpyFunc, couplingConstant)

computeMFCoupling = lambda couplingConstant, lambdaHam, phiFieldInit, voltageValue, TValue, ImpyFunc: computeMeanField(lambdaHam, phiFieldInit, voltageValue, TValue, ImpyFunc, couplingConstant)