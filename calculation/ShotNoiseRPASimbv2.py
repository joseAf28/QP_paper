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


Tvalue_ref = 1e-5

##? Auxiliar functions
def Impy(z, zp, beta, mu): 
    arrayZ = [(cmath.pi - 1j*beta*(z - mu)*np.sign(np.imag(z)))/(2*cmath.pi)]
    arrayZp = [(cmath.pi - 1j*beta*(zp - mu)*np.sign(np.imag(zp)))/(2*cmath.pi)]
        
    valueAux = (-2.0*scipy.special.psi(arrayZ) + 2.0*scipy.special.psi(arrayZp))/(2.0*cmath.pi*(z - zp))
    return valueAux


##! Leads Green function: Left Junction and considering the T -> 0 limit for now
def greenLeadsAdvanced(omega, couplingValue):
    return 0.5j*couplingValue

def greenLeadsRetarded(omega, couplingValue):
    return -0.5j*couplingValue

def greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue = 1e-5):
    beta2 = 0.5/Tvalue
    
    # return -1j*couplingValue*np.sign(omega - voltageValue/2.0)
    
    if Tvalue < Tvalue_ref:
        return -1j*couplingValue*np.sign(omega - voltageValue/2.0)
    else:
        return -1j*couplingValue*np.tanh(beta2*(omega - voltageValue/2.0))


##! Saddle Point Green function: Retarded, Advanced and Keldysh and considering the T -> 0 limit for now
def greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue):
    den = (omega - epsilon + 2.0*lambdaValue*phiValue) - 1j*couplingValue
    return np.reciprocal(den)

def greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue):
    den = (omega - epsilon + 2.0*lambdaValue*phiValue) + 1j*couplingValue
    return np.reciprocal(den)


def greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue=1e-5):
    beta = 1.0/Tvalue
    
    if Tvalue < Tvalue_ref:
        num = -1j*couplingValue*(np.sign(omega - voltageValue/2.0) + np.sign(omega + voltageValue/2.0))
    else:
        num = -1j*couplingValue*(np.tanh(0.5*beta*(omega - voltageValue/2.0)) + np.tanh(0.5*beta*(omega + voltageValue/2.0)))
    
    # num = -1j*couplingValue*(np.sign(omega - voltageValue/2.0) + np.sign(omega + voltageValue/2.0))
    
    den = (omega - epsilon + 2.0*lambdaValue*phiValue)**2 + couplingValue**2
    return num*np.reciprocal(den)





##? Shot Noise 
###! Non Interacting Keldysh Component


def gK_GK(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    def Sintegrand(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux01 = gK(nu3)*GK(nu2)
        
        return Saux01
    
    SintReal = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).real, -np.inf, np.inf, limit=200)
    SintImag = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).imag, -np.inf, np.inf, limit=200)
    
    return (SintReal[0] + 1j*SintImag[0])/(2.0*np.pi)



def gKGA_GRgK(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    voltage2 = voltageValue/2.0
    beta = 1.0/Tvalue
    omega2 = omega/2.0
    coupling2 = 1j*couplingValue/2.0
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
        
    def Sintegrand(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
    
        Saux = gK(nu3)*GA(nu2) + GR(nu3)*gK(nu2) + GK(nu3)*gA(nu2) + gR(nu3)*GK(nu2)
            
        return Saux

    SintReal = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    return (SintReal[0] + 1j*SintImag[0])/(4.0*np.pi)




def S01NonIntKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    ## (GA.gR + GR.gA)/(2\pi) = \Gamma/2
    value0 = couplingValue
    
    ## gK(nu+).GK(nu-)
    value1 = gK_GK(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)

    value2 = gK_GK(epsilon, -omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)

    result = value0 + (value1 + value2)/2.0
    return result



def S01NonIntRetarded(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    ### gK.GA + GR.gK
    value1 = 2*gKGA_GRgK(epsilon, -omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    return value1




#### Second Term Keldysh
def KeldyshNonIntOrder2(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):

    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux = gA(nu2)*GA(nu2)*gA(nu2)*GR(nu3) + gK(nu2)*GA(nu2)*gA(nu2)*GK(nu3) +\
                gR(nu2)*GK(nu2)*gA(nu2)*GK(nu3) + gR(nu2)*GR(nu2)*gK(nu2)*GK(nu3) +\
                gR(nu2)*GR(nu2)*gR(nu2)*GA(nu3)

        return Saux
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
    
        Saux = gA(nu2)*GA(nu2)*gR(nu3)*GR(nu3) + gK(nu2)*GA(nu2)*gK(nu3)*GA(nu3) +\
            gK(nu2)*GA(nu2)*gR(nu3)*GK(nu3) + gR(nu2)*GK(nu2)*gK(nu3)*GA(nu3) + \
            gR(nu2)*GK(nu2)*gR(nu3)*GK(nu3) + gR(nu2)*GR(nu2)*gA(nu3)*GA(nu3)

        return Saux
    
    
    def Sintegrand4(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
    
        Saux = GA(nu2)*gA(nu2)*GR(nu3)*gR(nu3) + GK(nu2)*gA(nu2)*GK(nu3)*gA(nu3) +\
            GK(nu2)*gA(nu2)*GR(nu3)*gK(nu3) + GR(nu2)*gK(nu2)*GK(nu3)*gA(nu3) + \
            GR(nu2)*gK(nu2)*GR(nu3)*gK(nu3) + GR(nu2)*gR(nu2)*GA(nu3)*gA(nu3)
            
        return Saux
    
    
    ### first part ok
    
    SintegrandPlus = lambda nu, omega: Sintegrand2(nu, omega) + Sintegrand2(nu, -omega)
    SintegrandMinus = lambda nu, omega: - Sintegrand3(nu, omega) - Sintegrand4(nu, omega)

    SintRealPlus = scipy.integrate.quad(lambda nu1: SintegrandPlus(nu1, omega).real, -np.inf, np.inf, limit=500)
    SintImagPlus = scipy.integrate.quad(lambda nu1: SintegrandPlus(nu1, omega).imag, -np.inf, np.inf, limit=500)
    
    SintRealMinus = scipy.integrate.quad(lambda nu1: SintegrandMinus(nu1, omega).real, -np.inf, np.inf, limit=500)
    SintImagMinus = scipy.integrate.quad(lambda nu1: SintegrandMinus(nu1, omega).imag, -np.inf, np.inf, limit=500)
    
    return (SintRealPlus[0] + SintRealMinus[0])  + 1j*(SintImagPlus[0] + SintImagMinus[0])



def RetardedNonIntOrder2(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):

    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux = gK(nu2)*GA(nu2)*gA(nu2)*GA(nu3) + gR(nu2)*GK(nu2)*gA(nu2)*GA(nu3) + \
                gR(nu2)*GR(nu2)*gK(nu2)*GA(nu3) + gR(nu2)*GR(nu2)*gR(nu2)*GK(nu3)
        
        return Saux
    
    
    def Sintegrand2_2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux = GK(nu2)*gA(nu3)*GA(nu3)*gA(nu3) + GR(nu2)*gK(nu3)*GA(nu3)*gA(nu3) + \
                GR(nu2)*gR(nu3)*GK(nu3)*gA(nu3) + GR(nu2)*gR(nu3)*GR(nu3)*gK(nu3)
        
        return Saux
    
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux = gK(nu2)*GA(nu2)*gA(nu3)*GA(nu3) + gR(nu2)*GK(nu2)*gA(nu3)*GA(nu3) + \
                gR(nu2)*GR(nu2)*gK(nu3)*GA(nu3) + gR(nu2)*GR(nu2)*gR(nu3)*GK(nu3)
        
        return Saux
    
    
    def Sintegrand4(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux = GK(nu2)*gA(nu2)*GA(nu3)*gA(nu3) + GR(nu2)*gK(nu2)*GA(nu3)*gA(nu3) + \
                GR(nu2)*gR(nu2)*GK(nu3)*gA(nu3) + GR(nu2)*gR(nu2)*GR(nu3)*gK(nu3)
        
        return Saux
    
    SintegrandPlus = lambda nu, omega: Sintegrand2(nu, omega) + Sintegrand2_2(nu, omega)
    SintegrandMinus = lambda nu, omega: - Sintegrand3(nu, omega) - Sintegrand4(nu, omega)

    SintRealPlus = scipy.integrate.quad(lambda nu1: SintegrandPlus(nu1, omega).real, -np.inf, np.inf, limit=1000)
    SintImagPlus = scipy.integrate.quad(lambda nu1: SintegrandPlus(nu1, omega).imag, -np.inf, np.inf, limit=1000)
    
    SintRealMinus = scipy.integrate.quad(lambda nu1: SintegrandMinus(nu1, omega).real, -np.inf, np.inf, limit=1000)
    SintImagMinus = scipy.integrate.quad(lambda nu1: SintegrandMinus(nu1, omega).imag, -np.inf, np.inf, limit=1000)
    
    return (SintRealPlus[0] + SintRealMinus[0]) + 1j*(SintImagPlus[0] + SintImagMinus[0])


def shotNoiseKeldyshNonInt(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue=const.Tvalue):

    value1 = S01NonIntKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    value2 = 0.5/(2.0*np.pi)*KeldyshNonIntOrder2(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)

    return value1 + value2


def shotNoiseRetardedNonInt(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue=const.Tvalue):
    
    value1 = S01NonIntRetarded(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    value2 = 1.0/(2.0*np.pi)*RetardedNonIntOrder2(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    return value1 + value2



def shotNoiseNonIntFCS(epsilon, lambdaValue, phiValue, voltageValue, couplingValue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue)

    GA = lambda epsilon, omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda epsilon, omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda epsilon, omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue)


    def g0mat(nu_var):
        return np.array([[gR(nu_var), gK(nu_var)], [0.0, gA(nu_var)]])
    
    def G0mat(epsilon_var, omega_var):
        return np.array([[GR(epsilon_var, omega_var), GK(epsilon_var, omega_var)], [0.0, GA(epsilon_var, omega_var)]])
    
    
    def Sigma1():
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    
    
    def MMat(nu_var):
        return 0.5*np.array([[-gA(nu_var) + gR(nu_var),  gK(nu_var)], [-gK(nu_var), gA(nu_var) - gR(nu_var)]])
    
    
    def integrand(nu1, epsilon):
        
        Sigma1Mat = Sigma1()
        
        S01 = np.dot(G0mat(epsilon, nu1), MMat(nu1))
        aux0 = -np.trace(S01)
        
        S02 = np.dot(np.dot(np.dot(np.dot(np.dot(G0mat(epsilon, nu1), MMat(nu1)), Sigma1Mat), G0mat(epsilon, nu1)), MMat(nu1)), Sigma1Mat)
        aux1 = np.trace(S02)
        
        return aux0 + aux1
        
    
    integrandReal = lambda nu1, epsilon: integrand(nu1, epsilon).real
    integrandImag = lambda nu1, epsilon: integrand(nu1, epsilon).imag
    
    SintReal = scipy.integrate.quad(integrandReal, -np.inf, np.inf, args=(epsilon), limit=1000)
    SintImag = scipy.integrate.quad(integrandImag, -np.inf, np.inf, args=(epsilon), limit=1000)
    
    
    return (SintReal[0] + 1j*SintImag[0])/(2.0*np.pi), SintReal[1], SintImag[1]
    




##? Shot Noise RPA
def M1q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux1 = GA(nu2)*GR(nu3)*gR(nu3) + GK(nu2)*GK(nu3)*gA(nu3) +\
                GK(nu2)*GR(nu3)*gK(nu3) + GR(nu2)*GA(nu3)*gA(nu3)

        return Saux1

    SintReal = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    return (SintReal[0] + 1j*SintImag[0])


### M2q term

def M2q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    def Sintegrand(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0

        Saux1 = GA(nu2)*gA(nu2)*GR(nu3) + GR(nu2)*gK(nu2)*GK(nu3) +\
                GK(nu2)*gA(nu2)*GK(nu3) + GR(nu2)*gR(nu2)*GA(nu3)

        return Saux1

    SintReal = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag = scipy.integrate.quad(lambda nu1: Sintegrand(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    return (SintReal[0] + 1j*SintImag[0])


### M1cl term

def M1cl(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand1(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 = GK(nu2)*GA(nu3)*gA(nu3)

        return Saux1
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 =  GR(nu2)*GK(nu3)*gA(nu3) 
        return Saux1
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 = GR(nu2)*GR(nu3)*gK(nu3)

        return Saux1
    
    
    SintReal1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).imag, -np.inf, np.inf, limit=100)

    SintReal3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    
    return (SintReal1[0] + 1j*SintImag1[0]) + (SintReal2[0] + 1j*SintImag2[0]) + (SintReal3[0] + 1j*SintImag3[0])


### M2q term

def M2cl(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand1(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 = gK(nu2)*GA(nu2)*GA(nu3)
        return Saux1
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 = gR(nu2)*GK(nu2)*GA(nu3)

        return Saux1
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
                
        Saux1 = gR(nu2)*GR(nu2)*GK(nu3)

        return Saux1


    SintReal1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    
    return (SintReal1[0] + 1j*SintImag1[0]) + (SintReal2[0] + 1j*SintImag2[0]) + (SintReal3[0] + 1j*SintImag3[0])


### K1q term

def K1q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand1(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux1 = GA(nu2)*GK(nu3)*gA(nu3)
        return Saux1
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux1 = GA(nu2)*GR(nu3)*gK(nu3)
        return Saux1
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0
        
        Saux1 = GK(nu2)*GR(nu3)*gR(nu3)
        return Saux1


    SintReal1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    
    return (SintReal1[0] + 1j*SintImag1[0]) + (SintReal2[0] + 1j*SintImag2[0]) + (SintReal3[0] + 1j*SintImag3[0])


### K2q term

def K2q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue):
    
    gA = lambda omega: greenLeadsAdvanced(omega, couplingValue)
    gR = lambda omega: greenLeadsRetarded(omega, couplingValue)
    gK = lambda omega: greenLeadsKeldysh(omega, couplingValue, voltageValue, Tvalue)
    
    GA = lambda omega: greenAdvanced(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GR = lambda omega: greenRetarded(epsilon, omega, lambdaValue, phiValue, couplingValue)
    GK = lambda omega: greenKeldysh(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
    
    
    def Sintegrand1(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0

        Saux1 = gA(nu2)*GA(nu2)*GK(nu3)
        return Saux1
    
    def Sintegrand2(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0

        Saux1 = gK(nu2)*GA(nu2)*GR(nu3)
        return Saux1
    
    def Sintegrand3(nu1, omega):
        nu2 = nu1 + omega/2.0
        nu3 = nu1 - omega/2.0

        Saux1 = gR(nu2)*GK(nu2)*GR(nu3)

        return Saux1

    SintReal1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag1 = scipy.integrate.quad(lambda nu1: Sintegrand1(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag2 = scipy.integrate.quad(lambda nu1: Sintegrand2(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    SintReal3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).real, -np.inf, np.inf, limit=100)
    SintImag3 = scipy.integrate.quad(lambda nu1: Sintegrand3(nu1, omega).imag, -np.inf, np.inf, limit=100)
    
    
    return (SintReal1[0] + 1j*SintImag1[0]) + (SintReal2[0] + 1j*SintImag2[0]) + (SintReal3[0] + 1j*SintImag3[0])




def computeMterms(epsilonSpace, omegaSpace, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue, pbar=None):

    m1cl = np.zeros(len(omegaSpace), dtype=complex)
    m2cl = np.zeros(len(omegaSpace), dtype=complex)

    m1q = np.zeros(len(omegaSpace), dtype=complex)
    m2q = np.zeros(len(omegaSpace), dtype=complex)

    k1q = np.zeros(len(omegaSpace), dtype=complex)
    k2q = np.zeros(len(omegaSpace), dtype=complex)
    
    for i in range(len(omegaSpace)):
        omega = omegaSpace[i]
        for epsilon in epsilonSpace:
        
            m1cl[i] += M1cl(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            m2cl[i] += M2cl(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            
            m1q[i] += M1q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            m2q[i] += M2q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            
            k1q[i] += K1q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            k2q[i] += K2q(epsilon, omega, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue)
            
            if pbar is not None:
                pbar.update(1)
    
    # pbar.close()
    
    m1cl = m1cl/len(epsilonSpace)
    m2cl = m2cl/len(epsilonSpace)
    m1q = m1q/len(epsilonSpace)
    m2q = m2q/len(epsilonSpace)
    k1q = k1q/len(epsilonSpace)
    k2q = k2q/len(epsilonSpace)
        
    
    Tcl = m1cl - m2cl
    Tq = m1q - m2q
    Kq = k1q - k2q
    
    return Tcl, Tq, Kq
    


def shotNoiseInt(DR, DK, epsilonSpace, omegaSpace, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue=const.Tvalue, pbar=None):
    
    
    Tcl, Tq, Kq = computeMterms(epsilonSpace, omegaSpace, lambdaValue, phiValue, voltageValue, couplingValue, Tvalue, pbar)
    
    TclFuncReal = scipy.interpolate.interp1d(omegaSpace, Tcl.real, kind='linear')
    TclFuncImag = scipy.interpolate.interp1d(omegaSpace, Tcl.imag, kind='linear')

    TqFuncReal = scipy.interpolate.interp1d(omegaSpace, Tq.real, kind='linear')
    TqFuncImag = scipy.interpolate.interp1d(omegaSpace, Tq.imag, kind='linear')

    KqFuncReal = scipy.interpolate.interp1d(omegaSpace, Kq.real, kind='linear')
    KqFuncImag = scipy.interpolate.interp1d(omegaSpace, Kq.imag, kind='linear')

    DRFuncReal = scipy.interpolate.interp1d(omegaSpace, DR.real, kind='linear')
    DRFuncImag = scipy.interpolate.interp1d(omegaSpace, DR.imag, kind='linear')

    DKFuncImag = scipy.interpolate.interp1d(omegaSpace, DK.imag, kind='linear')

    TclFunc = lambda omega: TclFuncReal(omega) + 1j*TclFuncImag(omega)
    TqFunc = lambda omega: TqFuncReal(omega) + 1j*TqFuncImag(omega)
    KqFunc = lambda omega: KqFuncReal(omega) + 1j*KqFuncImag(omega)
    
    DRFunc = lambda omega: DRFuncReal(omega) + 1j*DRFuncImag(omega)
    DAFunc = lambda omega: DRFuncReal(omega) - 1j*DRFuncImag(omega)

    DKFunc = lambda omega: 1j*DKFuncImag(omega)
    
    
    def SRfunc(omega, lambdaValue):
        value = DRFunc(omega)*TclFunc(omega)*KqFunc(-omega)
        return 2.0*lambdaValue**2*value/(2.0*np.pi)


    def SKfunc(omega, lambdaValue):
        ###! check missing 2
        DK = 2*DKFunc(omega)
        value1 = DK*TclFunc(omega)*TclFunc(-omega)
        value2 = DRFunc(omega)*TclFunc(omega)*TqFunc(-omega)
        value3 = DRFunc(-omega)*TclFunc(-omega)*TqFunc(omega)
        
        return 2.0*lambdaValue**2*(value1 + value2 + value3)/(2.0*np.pi)
    
    
    SRvec = np.zeros(len(omegaSpace), dtype=complex)
    SKvec = np.zeros(len(omegaSpace), dtype=complex)
    
    for i in range(len(omegaSpace)):
        omega = omegaSpace[i]
        
        valueSR = SRfunc(omega, lambdaValue)
        valueSK = SKfunc(omega, lambdaValue)
        
        SRvec[i] = SRfunc(omega, lambdaValue)
        SKvec[i] = SKfunc(omega, lambdaValue)
    
    
    return SKvec, SRvec