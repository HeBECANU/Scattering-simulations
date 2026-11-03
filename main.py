from core import *
from plotting import *
from tests import *

'''
How to use this simulation

This simulation has three main components
1) Energy evaluation
2) Collision evaluation
3) Scattering simulation
'''
#Energy evaluation example
#To evaluate energies, we first choose a trapping frequency in MHz and atom numbers for both He3 and He4
#a non-zero temperature (in K) can also be submitted, though the variation between non-zero temperature and zero-temperature
#density profiles is fairly insignificant and therefore zero-temperature density profiles typically suffice
# fx, fy, fz = 50e-6, 500e-6, 501e-6
# Nb, Nf = 30000, 10000
# T = 0
# mu_f, mu_b = evaluateEnergies(fx, fy, fz, T, Nf, Nb)

#Collision evaluation example
#Now that energies have been evaluated, a Bose-Bose collision and a Bose-Fermi collision can be evaluated
#In both collision functions, tfree indicates the time that each atomic cloud is allowed to expand before the momentum kick
#In the bose-bose collision, the transfer fraction can be adjusted with the additional parameter 'transfer'
# rate, time, collisions, err = evaluateCollisions(mu_f, mu_b, fx, fy, fz, T, Nf, tfree=1200) #bose-fermi collision
# rate, time, collisions = boseBoseCollision(mu_b, fx, fy, fz, tfree=1200) #bose-bose collision

#Scattering simulation example
#Here, fermionDetection and bosonDetection are floats between 0 and 1 which indicate the fraction of particles which
#land on the detector in the Bose-Fermi scattering case
# XYT_detected, species_detected, fermionDetection, bosonDetection, collisions = evaluateDetectorImageXY(fx, fy, fz, T, Nb+Nf, Nb/Nf)


# fx, fy, fz = 21.037e-6, 44.325e-6, 46.940e-6
# COLLISIONS = []
# for i in range(25):
#     print(i)
#     F, T = 0, 0

#     while F <= 0 or F>= 1 or T<=0 or T>= 1:
#         N, F, T = drawNumberFractionTransfer(10100, 600, 0.55, 0.04, 0.52, 0.07)
#     mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)

#     cr, T, collisions = boseBoseCollision(mu_b, fx, fy, fz, tfree=1200, vkick = 0.128, transfer =T, fraction=F)
#     COLLISIONS.append(collisions)
# print(np.average(COLLISIONS)*2, np.std(COLLISIONS)*2)


# fx, fy, fz = 31.88e-6, 84.44e-6, 88.76e-6
# COLLISIONS = []
# for i in range(25):
#     print(i)
#     F, T = 0, 0
#     while F <= 0 or F>= 1 or T<=0 or T>= 1:
#         N, F, T = drawNumberFractionTransfer(12600, 400, 0.71, 0.06, 0.4, 0.1)
#     mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)

#     cr, T, collisions = boseBoseCollision(mu_b, fx, fy, fz, tfree=1200, vkick = 0.128, transfer =T, fraction=F)
#     COLLISIONS.append(collisions)
# print(np.average(COLLISIONS)*2, np.std(COLLISIONS)*2)


# fx, fy, fz = 41.09e-6, 115.2e-6, 120.65e-6
# COLLISIONS = []
# for i in range(25):
#     print(i)
#     F, T = 0, 0

#     while F <= 0 or F>= 1 or T<=0 or T>= 1:
#         N, F, T = drawNumberFractionTransfer(30000, 1000, 0.66, 0.06, 0.44, 0.09)
#     mu_f, mu_b = evaluateEnergies(fx, fy, fz, 0, 0, N)

#     cr, T, collisions = boseBoseCollision(mu_b, fx, fy, fz, tfree=1200, vkick = 0.128, transfer =T, fraction=F)
#     COLLISIONS.append(collisions)
# print(np.average(COLLISIONS)*2, np.std(COLLISIONS)*2)

# fx, fy, fz = 21.037e-6, 44.325e-6, 46.940e-6
# N = 10100
# T_c = evaluateCriticalTemperature(fx, fy, fz, N, 600)
# print(T_c)
# fx, fy, fz = 31.88e-6, 84.44e-6, 88.76e-6
# N = 12600
# T_c = evaluateCriticalTemperature(fx, fy, fz, N, 400)
# print(T_c)
# fx, fy, fz = 41.09e-6, 115.2e-6, 120.65e-6
# N = 30000
# T_c = evaluateCriticalTemperature(fx, fy, fz, N, 1000)
# print(T_c)

nf = 10000
nb = 30000
fx, fy, fz = tight[0], tight[1], tight[2]

# evaluateDetectorImageXY(ultraweak[0], ultraweak[1], ultraweak[2], 0.6, nf+nb, nb/nf)
evaluateDetectorImageXY(tight[0], tight[1], tight[2], 0.6, nf+nb, nb/nf)
# plotDetectionFraction(0.6, 3)