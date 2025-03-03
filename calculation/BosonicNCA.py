import numpy as np
import scipy
import h5py
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import sys
import cmath
import vegas
import itertools
import numba

script_dir = os.path.dirname(__file__)
settings_path = os.path.join(script_dir, '..', 'settings')
settings_path = os.path.abspath(settings_path)
sys.path.append(settings_path)
calculation_path = os.path.join(script_dir, '..', 'calculation')
calculation_path = os.path.abspath(calculation_path)
sys.path.append(calculation_path)

import MeanFieldv2 as MFv2
import Constants as const
import SusceptibilitySimpv2 as SusceptibilitySimp


DOSoperations = {'linear': MFv2.DOS1func, 'lorentzian': MFv2.DOS2func, 'gaussianMod': MFv2.DOS3func, 'bilevel': MFv2.DOS4func}


##! Saddle Point Green function: Retarded, Advanced and Keldysh and considering the T -> 0 limit for now
def greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue):
    den = (omega - epsilon + 2.0*lambdaValue*phiValue) - 1j*couplingValue
    return np.reciprocal(den)

def greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue):
    den = (omega - epsilon + 2.0*lambdaValue*phiValue) + 1j*couplingValue
    return np.reciprocal(den)

def greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue):
    num = -1j*couplingValue*(np.sign(omega - voltageValue/2.0) + np.sign(omega + voltageValue/2.0))
    den = (omega - epsilon + 2.0*lambdaValue*phiValue)**2 + couplingValue**2
    return num*np.reciprocal(den)


##! NCA's Self Energy functions
def AdvancedSelfEnergy(epsilon, omegaValue, phiValue, Tvalue, lambdaConstant, voltageValue, couplingConstant, nuSpace, DA, DK):
    beta = 1.0/const.Tvalue
    invLambda = 1.0/lambdaConstant
    
    ## Advanced Green Functions
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaConstant, phiValue, couplingConstant)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaConstant, phiValue, voltageValue, couplingConstant)
    
    ## Bosonic Green Functions
    DARealInter = scipy.interpolate.interp1d(nuSpace, np.real(DA), kind='linear', fill_value=np.real(DA[-1]), bounds_error=False)
    DAImagInter = scipy.interpolate.interp1d(nuSpace, np.imag(DA), kind='linear', fill_value=0.0, bounds_error=False)
    
    DKImagInter = scipy.interpolate.interp1d(nuSpace, np.imag(DK), kind='linear', fill_value=0.0, bounds_error=False)
    
    
    def integrandAdvancedSelfEnergy(nu1, omega):
        omega2 = omega/2.0
        
        nu1plus = nu1
        nu1minus = -nu1 + omega
        
        advancedGreenAux = GA(nu1minus)
        keldyshGreenAux = GK(nu1minus)
        
        DAAux = DARealInter(nu1plus) + 1j*DAImagInter(nu1plus)
        DKAux = 1j*DKImagInter(nu1plus)
        
        return advancedGreenAux*DKAux + keldyshGreenAux*DAAux
    
    omegaMax = nuSpace[-1]
    
    integrandAdvancedSelfEnergyReal = lambda nu1, omega: np.real(integrandAdvancedSelfEnergy(nu1, omega))
    integrandAdvancedSelfEnergyImag = lambda nu1, omega: np.imag(integrandAdvancedSelfEnergy(nu1, omega))
    
    integrateAdvancedSelfEnergyReal = lambda omega: scipy.integrate.quad(integrandAdvancedSelfEnergyReal, -omegaMax, omegaMax, args=(omega), limit=90)[0]
    integrateAdvancedSelfEnergyImag = lambda omega: scipy.integrate.quad(integrandAdvancedSelfEnergyImag, -omegaMax, omegaMax, args=(omega), limit=90)[0]
    
    integrateAdvancedSelfEnergy = lambda omega: integrateAdvancedSelfEnergyReal(omega) + 1j*integrateAdvancedSelfEnergyImag(omega)
    
    advancedSelfEnergy = integrateAdvancedSelfEnergy(omegaValue)

    return 4.0*lambdaConstant**2*1j*advancedSelfEnergy/(2.0*cmath.pi)


def KeldyshSelfEnergy(epsilon, omegaValue, phiValue, Tvalue, lambdaConstant, voltageValue, couplingConstant, nuSpace, DA, DK):
    beta = 1.0/const.Tvalue
    invLambda = 1.0/lambdaConstant
    
    ## Saddle Point Green Functions
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaConstant, phiValue, couplingConstant)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaConstant, phiValue, voltageValue, couplingConstant)
    
    ## Bosonic Green Functions
    DARealInter = scipy.interpolate.interp1d(nuSpace, np.real(DA), kind='linear', fill_value=np.real(DA[-1]), bounds_error=False)
    DAImagInter = scipy.interpolate.interp1d(nuSpace, np.imag(DA), kind='linear', fill_value=0.0, bounds_error=False)
    
    DKImagInter = scipy.interpolate.interp1d(nuSpace, -np.imag(DK), kind='linear', fill_value=0.0, bounds_error=False)
    
    
    def integrandSelfEnergy1(nu1, omega):
        
        nu1plus = nu1
        nu1minus = -nu1 + omega
        
        keldyshGreenAux = GK(nu1minus)
        DKAux = 1j*DKImagInter(nu1plus)
        
        return keldyshGreenAux*DKAux
    
    
    def integrandSelfEnergy2(nu1, omega):
        
        nu1plus = nu1
        nu1minus = -nu1 + omega
        
        retardedGreenAux = np.conj(GA(nu1minus))
        DRAux = DARealInter(nu1plus) - 1j*DAImagInter(nu1plus)
        
        return 2.0*np.real(retardedGreenAux*DRAux)

    
    omegaMax = 1e5
    
    integrandSelfEnergyReal1 = lambda nu1, omega: np.real(integrandSelfEnergy1(nu1, omega))
    integrandSelfEnergyReal2 = lambda nu1, omega: np.real(integrandSelfEnergy2(nu1, omega))
    
    # integrandSelfEnergyImag = lambda nu1, omega: np.imag(integrandSelfEnergy(nu1, omega))
    
    integrateSelfEnergyReal = lambda omega: scipy.integrate.quad(integrandSelfEnergyReal1, -omegaMax, omegaMax, args=(omega), limit=100)[0]\
                                        + scipy.integrate.quad(integrandSelfEnergyReal2, -omegaMax, omegaMax, args=(omega), limit=100)[0]                    
    # integrateSelfEnergyImag = lambda omega: scipy.integrate.quad(integrandSelfEnergyImag1, -omegaMax, omegaMax, args=(omega), limit=100)[0]
    
    integrateSelfEnergy = lambda omega: integrateSelfEnergyReal(omega)
    
    keldyshSelfEnergy = integrateSelfEnergy(omegaValue)

    return 4.0*lambdaConstant**2*1j*keldyshSelfEnergy/(2.0*cmath.pi)
