import os
import numpy as np
##? This file contains all the constants and default values used in the MeanField and Calc scripts  

##! Files for the output

saddlePointFile = "results.h5"
saddlePointFileTvalue = "resultsT.h5"

shotNoiseFile = "resultsShotNoiseNearZero.h5"
shotNoiseZeroFile = "resultsShotNoiseZero-2.h5"

shotNoiseFileTvalue = "resultsShotNoiseTvalue-2.h5"
shotNoiseFileZeroTvalue = "resultsShotNoiseZeroV2Tvalue2-2.h5"


##! Computational options for MeanField self consistent calculation
modeFindRoot = 'lambda' # 'phi', 'lambda' 

##! Energy Levels and DOS functions for the Quantum Dot
DOSfunc = 'uniform' # 'uniform',  'lorentzian',  'gaussianMod', 'levels' 


##* Default values for the functions. They are overwritten by the values in the .ipynb files

LambdaDOS = 2.0     # Lorentzian DOS
nuDOS = 1           # Gaussian DOS
sigmaDOS = 1.0      # Gaussian DOS
epsilonDOS = 1.0    # Bilevel DOS and Linear DOS - 0.5 seems better than 1.0

nbDOS = 10

couplingConstant = 0.7 # Default value for the coupling constant

phiConstantInit = 0.9
phiLambdaInit = 1.0

Tvalue = 1e-5   # Default value for the temperature

TvalueRPA = Tvalue
TvalueShotNoiseRPA = Tvalue



####* HyperParameters for the Shot Noise calculation

nb_cores = 9
nbDOS_shotNoise = 100 # or 100


omegaSpaceZero = np.linspace(-1e-3, 1e-3, 2)

# voltageSpaceL = np.linspace(1e-4, 1.25, 45)
# voltageSpaceC = np.linspace(1.25, 1.55, 20)
# voltageSpaceR = np.linspace(1.55, 2.0, 10)
# voltageSpaceZero = np.unique(np.concatenate((voltageSpaceL, voltageSpaceC, voltageSpaceR)))

voltageSpaceZero = np.linspace(1e-4, 1.9, 150)


# omegaSpaceLeft =np.linspace(-6.0, -2.5, 8)
# omegaSpaceRight = np.linspace(2.5, 6.0, 8)

# omegaSpaceCenterL = np.linspace(-2.5, -1.1e-2, 30)
# omegaSpaceCenterC = np.linspace(-1e-2, 1e-2, 2)
# omegaSpaceCenterR = np.linspace(1.1e-2, 2.5, 30)
# omegaSpaceCenter = np.unique(np.concatenate([omegaSpaceCenterL, omegaSpaceCenterC, omegaSpaceCenterR]))

# omegaSpace = np.unique(np.concatenate([omegaSpaceLeft, omegaSpaceCenter, omegaSpaceRight]))



# omegaSpaceLeft =np.linspace(-6.0, -2.5, 10)
# omegaSpaceRight = np.linspace(2.5, 6.0, 10)

# # omegaSpaceCenterL = np.linspace(-2.5, -1.5e-2, 20)
# # omegaSpaceCenterC = np.linspace(-1e-3, 1e-3, 2)
# # omegaSpaceCenterR = np.linspace(1.5e-2, 2.5, 20)
# # omegaSpaceCenter = np.unique(np.concatenate([omegaSpaceCenterL, omegaSpaceCenterC, omegaSpaceCenterR]))

# omegaSpaceCenter = np.linspace(-2.5, 2.5, 41)


# omegaSpace = np.unique(np.concatenate([omegaSpaceLeft, omegaSpaceCenter, omegaSpaceRight]))

omegaSpace = np.linspace(-1e-3, 1e-3, 2)

# omegaSpace = np.linspace(-1e-3, 1e-3, 41)


Tvalue_ref = 5e-5



