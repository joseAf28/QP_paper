import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

##! Auxiliary functions to create the many-body space

def andStates(state1, state2, M):
    stringAux = ''
    for i in range(M):
        boolValue = state1[i] == state2[i]
        if boolValue:
            stringAux = stringAux + state1[i]
        else:
            stringAux = stringAux + '0'
    return str(stringAux)


def orStates(state1, state2, M):
    stringAux = ''
    for i in range(M):
        bool1 = state1[i] == '0'
        bool2 = state2[i] == '0'
        if bool1 and bool2:
            stringAux = stringAux + '0'
        else:
            stringAux = stringAux + '1'
    return str(stringAux)


def normState(state, M):
    if len(state) > M:
        print('Watch out! State has more sites than the system size')
        return state
    elif len(state) < M:
        state = state + '0'* (M - len(state))
        return state
    else:
        return state


def addState(state, i, M):
    stateAux = '0'*(i-1) + '1' + '0'*(M-i)
    return orStates(state, stateAux, M)


def removeState(state, i, M):
    stateAux = '0'*(i-1) + '1' + '0'*(M-i)
    return andStates(state, stateAux, M)


def int2State(number, M):
    stateAux = bin(number)[2:]
    return normState(stateAux[::-1], M)

def state2Int(state, M):
    return int(state[::-1], 2)


def createState(idxVec, M):
    state = ''
    for i in range(M):
        if i in idxVec:
            state = state + '1'
        else:
            state = state + '0'
    return state


##! Second quantization operators
def dOperator(state, idx):
    if idx >= len(state):
        return 0, state
    else:
        ## Annihilate a fermion at site i in the given state.
        if not state[idx] == '1':
            return 0, state  # Return zero if the site is unoccupied
        
        sign = (-1)**(state[:idx].count('1'))
        return sign, state[:idx] + '0' + state[idx+1:]


def dDaggerOperator(state, idx):
    if idx >= len(state):
        return 0, state
    else:
        ## Create a fermion at site i in the given state.
        if state[idx] == '1':
            return 0, state  # Return zero if the site is already occupied
        else:
            sign = (-1)**(state[:idx].count('1'))
            return sign, state[:idx] + '1' + state[idx+1:]


##! Create Lindblad matrix
def lindbladSuperoperator(L, Ldag, identity, gamma):
    
    Ldiss1 = np.zeros_like(L[0])
    Ldiss2 = np.zeros_like(L[0])
    Ldiss3 = np.zeros_like(L[0])
    
    Ldiss1 = np.kron(np.conj(L[0]), L[0]) + np.kron(np.conj(L[1]), L[1])
    Ldiss2 = -0.5 * np.kron(identity, np.dot(Ldag[0], L[0])) - 0.5 * np.kron(identity, np.dot(Ldag[1], L[1]))
    Ldiss3 = -0.5 * np.kron(np.transpose(np.dot(Ldag[0], L[0])), identity) - 0.5 * np.kron(np.transpose(np.dot(Ldag[1], L[1])), identity)
    
    return gamma * (Ldiss1 + Ldiss2 + Ldiss3)


##! Quantum Jump Operators
def quantumJumpOperators(M, alpha):
    Lleft = np.zeros((2**M,2**M), dtype=np.float64)
    Lright = np.zeros((2**M,2**M), dtype=np.float64)

    for state in range(2**M):
        stateBin = int2State(state, M)
            
        for state2 in range(2**M):
            stateBin2 = int2State(state2, M)
                
            sign, new_state = dOperator(stateBin2, alpha)
            sign2, new_state2 = dDaggerOperator(stateBin2, alpha)
                
            if stateBin == new_state:
                Lleft[state, state2] += sign
                
            if stateBin == new_state2:
                Lright[state, state2] += sign2
        
    return Lleft, Lright


##! Fill the Hamiltonian matrix of the Non-Interacting Case
def hamiltonianNonIntAlpha(M, epsilon, lambdaValue):
    
    HMat = np.zeros((2**M,2**M), dtype=np.float64)

    ##! Fill the Hamiltonian matrix
    for state in range(2**M):
        stateBin = int2State(state, M)
        
        for state2 in range(2**M):
            stateBin2 = int2State(state2, M)
                
            for alpha in range(M): # energy levels dot term
            # energy levels dot term
                sign, new_state = dOperator(stateBin2, alpha)
                sign2, new_state2 = dDaggerOperator(new_state, alpha)
                    
                if stateBin == new_state2:
                    HMat[state, state2] += epsilon[alpha]*float(sign * sign2)

    return HMat



##! Fill the Hamiltonian matrix of the Interacting Case
def hamiltonianInt2Alpha(M, epsilonVec, lambdaValue):
    HMat = np.zeros((2**M,2**M), dtype=np.float64)

    ##! Fill the Hamiltonian matrix
    for state in range(2**M):
        stateBin = int2State(state, M)
        
        for state2 in range(2**M):
            
            stateBin2 = int2State(state2, M)
            
            for alpha in range(M):
                sign, new_state = dOperator(stateBin2, alpha)
                sign2, new_state2 = dDaggerOperator(new_state, alpha)
                    
                if stateBin == new_state2:
                    HMat[state, state2] += (epsilonVec[alpha] + lambdaValue)*float(sign * sign2)
            
            for alpha in range(M):
                for j in range(M):

                    sign3, new_state3 = dOperator(stateBin2, j)
                    sign4, new_state4 = dDaggerOperator(new_state3, j)
                    sign5, new_state5 = dOperator(new_state4, alpha)
                    sign6, new_state6 = dDaggerOperator(new_state5, alpha)

                    if stateBin == new_state6:
                        HMat[state, state2] -= (lambdaValue/M)*float(sign3 * sign4 * sign5 * sign6)
            
            if state == state2:
                HMat[state, state2] -= 0.25*lambdaValue*M
    
    return HMat

##! Shot Noise Calculation using the Lindblad formalism
def shotNoiseLinbladian(funcHamiltonian, M, epsilonSpace, lambdaValue, gamma, omegaVec):

    ##! Create the matrix representation of the operators and of the Lindblad
    identity = np.eye(2**M)
    
    HMatM = np.zeros((2**M,2**M), dtype=np.float64)
    
    LleftM = np.zeros((M, 2**M,2**M), dtype=np.float64)
    LrightM = np.zeros((M, 2**M,2**M), dtype=np.float64)
    
    lindbladMatrixIntM = np.zeros((2**(2*M),2**(2*M)), dtype=np.complex128)
    for i in range(M):
        LleftM[i, :, :], LrightM[i, :, :] = quantumJumpOperators(M, i)
        
        lindbladMatrixIntM += lindbladSuperoperator([LleftM[i], LrightM[i]], [np.conj(LleftM[i].T), np.conj(LrightM[i].T)], identity, gamma)
    
    Hmat = funcHamiltonian(M, epsilonSpace, lambdaValue)
    HmatLindblad = -1j * np.kron(identity, Hmat) + 1j * np.kron(np.transpose(Hmat), identity)
    
    lindbladMatrixIntM += HmatLindblad

    ##! Diagonalize the Lindblad matrix
    evalsNonInt, evecsLNonInt, evecsRNonInt = scipy.linalg.eig(lindbladMatrixIntM, left=True, right=True)
    
    print("evalsNonInt: ", evalsNonInt)
    
    ##! Normalization Eigenstates
    for i in range(len(evalsNonInt)):
        norm = np.dot(np.conj(evecsLNonInt[:, i]), evecsRNonInt[:, i])
        evecsLNonInt[:, i] /= np.sqrt(norm)
        evecsRNonInt[:, i] /= np.sqrt(norm)


    idxZero = np.argmin(np.abs(evalsNonInt))
    idxEigen = np.arange(evalsNonInt.shape[0])
    idxEigen = np.delete(idxEigen, idxZero)

    evecsLModNonInt = evecsLNonInt[:, idxEigen]
    evecsRModNonInt = evecsRNonInt[:, idxEigen]
    evalsModNonInt = evalsNonInt[idxEigen]

    rho_ss_vecNonInt = evecsRNonInt[:, idxZero]
    one_L_vecNonInt = evecsLNonInt[:, idxZero]

    rho_ssNonInt = np.reshape(rho_ss_vecNonInt, (2**M, 2**M))
    rho_ssNonInt = rho_ssNonInt / np.trace(rho_ssNonInt)


    ##! Current Super Operator and Mean Current
    Jvec = np.zeros((2**(2*M), 2**(2*M)), dtype=np.complex128)
    
    for i in range(M):
        Jvec += gamma*np.kron(np.conj(LleftM[i]), LleftM[i])

    current = np.dot(one_L_vecNonInt,np.dot(Jvec, rho_ss_vecNonInt))/M
    
    if np.abs(current - 0.5*gamma) > 1e-3:
        print('Current is not equal to gamma')
        print(current, gamma)
        print('Error')
        return None

    ##! K superoperator
    K = 0.0

    K = np.zeros((2**(2*M), 2**(2*M)), dtype=np.complex128)
    for i in range(M):
        # K += gamma**2*np.trace(np.dot(np.dot(Ldagvec[i],Lvec[i]), rho_ssNonInt))
        K += gamma**2*np.trace(np.dot(np.dot(np.conj(LleftM[i].T),LleftM[i]), rho_ssNonInt))

    K = np.real(K)/2.0
    ##? Not included for now ...
    K = 0.0
    
    ##! Shot Noise Calculation
    
    ##? optimized version
    JvecRhoS1 = np.dot(Jvec, rho_ss_vecNonInt)
    
    def shotNoise1(omega):
        SauxValue = 0.0
        for i in range(evalsModNonInt.shape[0]):
            coef = evalsModNonInt[i]
            coefS = coef/(coef**2 + omega**2)
            SauxValue += coefS*np.dot(one_L_vecNonInt ,np.dot(Jvec, evecsRModNonInt[:, i]))*np.dot(evecsLModNonInt[:, i], JvecRhoS1)
            
        return -2.0*SauxValue
    
    # S1Aux = np.dot(evecsLModNonInt.T, JvecRhoS1)
    
    # JvecMatS2 = np.dot(Jvec, evecsRModNonInt)
    # S2Aux = np.dot(one_L_vecNonInt, JvecMatS2)
    
    # def shotNoise2(omega):
    #     coefVec = evalsModNonInt*np.reciprocal(evalsModNonInt**2 + omega**2)
    #     value = -2.0*coefVec*S1Aux*S2Aux
    #     return np.sum(value)


    SVec = np.zeros(omegaVec.shape[0], dtype=complex)
    
    pbar = tqdm(total=omegaVec.shape[0])
    for i in range(omegaVec.shape[0]):
        SVec[i] = shotNoise1(omegaVec[i])
        pbar.update(1)
        
    pbar.close()
    
    SVecFinal = SVec/M + K/M
    return -SVecFinal, current



##! General Parameters
# lambdaValue = 20.9001
lambdaValue = 1.0
gamma = 0.60
M = 4 ## Number of channels - each channel has an energy level in the dot

omegaVec = np.linspace(-8.0, 8.0, 300)

epsilonVec = np.linspace(-1.0, 1.0, M)
# epsilonVec = np.ones(M)*0.0


##! Non-Interacting Case Parameters
print("Non-Interacting Case")

shotNonInt = np.zeros((omegaVec.shape[0]), dtype=complex)
currentNonInt = 0.0

check = True
while check:
    result = shotNoiseLinbladian(hamiltonianNonIntAlpha, M, epsilonVec, lambdaValue, gamma, omegaVec)
    if result is not None:
        check = False
        shotNonInt = result[0]
        currentNonInt = result[1]
    else:
        epsilonVec = epsilonVec + 1e-5*np.random.randn(M)
        print("Failed to compute the current:")
        print("epsilonSpace: ", epsilonVec)
        print()
    
print("Current Non-Interacting: ", currentNonInt)

plt.figure(1)
plt.plot(omegaVec, np.real(shotNonInt), label="Non-Interacting Re")
plt.plot(omegaVec, np.imag(shotNonInt), label="Non-Interacting Im")

plt.xlabel(r'$\omega$')
# plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('ShotNoiseNonInt'+ str(M) +'ones0.png')


##! Interacting Case Parameters
print()
print("Interacting Case")

shotInt = np.zeros((omegaVec.shape[0]), dtype=complex)
currentInt = 0.0

check = True
while check:
    result = shotNoiseLinbladian(hamiltonianInt2Alpha, M, epsilonVec, lambdaValue, gamma, omegaVec)
    if result is not None:
        check = False
        shotInt = result[0]
        currentInt = result[1]
    else:
        epsilonVec = epsilonVec + 1e-5*np.random.randn(M)
        print("Failed to compute the current:")
        print("epsilonSpace: ", epsilonVec)
        print()
    
print("Current Interacting: ", currentInt)

plt.figure(2)
plt.plot(omegaVec, np.real(shotInt), label="Interacting Re")
plt.plot(omegaVec, np.imag(shotInt), label="Interacting Im")
plt.plot(omegaVec, np.real(shotNonInt), label="Non-Interacting Re")
plt.plot(omegaVec, np.imag(shotNonInt), label="Non-Interacting Im")

plt.xlabel(r'$\omega$')
# plt.yscale('log')
plt.legend()
plt.grid()
plt.savefig('ShotNoiseInt'+ str(M) +'ones0.png')