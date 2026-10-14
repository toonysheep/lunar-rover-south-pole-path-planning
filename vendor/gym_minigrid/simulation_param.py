import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import os
import statistics
from numpy.linalg import multi_dot

map_reso = 5 # speficy the map resolution of the used DEM data
slopeScaler = 1
mapsize = [200, 200]#[100,100]
location = [2010,1690]#[100,600][0,0][1600,1600]

xlength = mapsize[0]
ylength = mapsize[1]
#xlength = 100
width = xlength+2
height = ylength+2
max_env_steps_for_graphmaking = 1440 
timePerStep = 30 # time step of thermal and power calculation (min)
landingLatitudeDeg = -85 # 45 # degree from equator negative -> south, positive -> north
#site01 89.4°S, 210.3°E
#site06 85.4°S, 35.3°E
#site07 88.8°S, 124.7°E
landingLongitudeDeg = 35 #0

sunInitialTheta = 90 # note: the definition of theta - degree from equator 即月球和太阳在xy平面上
sunInitialPhi = 30

lunarRotationCycle = 29.5*24*60 # min
P_s = 1370 # Solar constant [W*m-2]
absorptivitySurface = 0.88 # Absorptivity of lunar surface
emissivitySurface = 0.94 # Emissivity of lunar surface
stephanBoltz = 5.67e-8 # σ : Stefan-Boltzmann constant 5.67×10^-8 [W/m^2/K^4]

# Calculate local Topocentric Frame with regard to Cartesian frame
localLat = landingLatitudeDeg
localLon = landingLongitudeDeg

# rotate around the current Z axis by localLongitude
rotationMat1 =  [[np.cos(np.deg2rad(localLon)), -np.sin(np.deg2rad(localLon)), 0],
                [np.sin(np.deg2rad(localLon)), np.cos(np.deg2rad(localLon)), 0],
                [0, 0, 1]]
# rotate around the current Y axis by localLattitude
rotationMat2 =  [[np.cos(np.deg2rad(-localLat)), 0, np.sin(np.deg2rad(-localLat))],
                [0, 1, 0],
                [-np.sin(np.deg2rad(-localLat)), 0, np.cos(np.deg2rad(-localLat))]]
# change the order of elements UEN -> NEU
rotationMat3 =  [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]]
                
# rotation matrix from XYZ to NEU
rotationXYZ2NEU = multi_dot([rotationMat1, rotationMat2, rotationMat3])

Xcart = [1, 0, 0]
Ycart = [0, 1, 0]
Zcart = [0, 0, 1]
N = np.dot(Xcart, rotationXYZ2NEU) # North vector with regard to CartX
E = np.dot(Ycart, rotationXYZ2NEU) # East vector with regard to CartY
U = np.dot(Zcart, rotationXYZ2NEU) # Up vector with regard to CartZ

TopoX = E
TopoY = N
TopoZ = U
