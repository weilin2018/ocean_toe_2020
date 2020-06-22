# -*- coding: utf-8 -*-

""" Use EN4 climatological volume of isopycnals to construct a density/pressure relationship used for remapping data from density to a pseudo-depth coordinate¶ """

import numpy as np
import matplotlib.pyplot as plt
import os, glob
from netCDF4 import Dataset as open_ncfile
import time

# ===
# === WORKSPACE AND PRE-REQUISITES ===
# ===

# -- Read EN4 file : will be our climatology; and reconstruct volume of isopycnals

# Read global thickness of isopycnals and compute volume (because wrong zonal volume)
f = open_ncfile('/data/ericglod/Density_binning/Obs_Prod_density_april16/obs.EN4.historical.r0i0p0.mo.ocn.Omon.density.ver-1.latestX.nc')
thick = np.ma.average(f.variables['isonthickg'][:,:,:,:],axis=0)/1.e03
targetGrid = '/home/ysilvy/Density_bining/Yona_analysis/data/170224_WOD13_masks.nc'
# Target horizonal grid
gridFile_f  = open_ncfile(targetGrid,'r')
maskg       = gridFile_f.variables['basinmask3'][:]
areai = gridFile_f.variables['basinmask3_area'][:] #(latitude,longitude), in km2
Nii     = areai.shape[1]
Nji     = areai.shape[0]
N_s = thick.shape[0]
areaisig = np.tile(np.ma.reshape(areai,Nii*Nji), (N_s,1))
areaisig = np.ma.reshape(areaisig,[N_s,Nji,Nii])
maski       = maskg.mask ; # Global mask
# Regional masks
maskAtl = maski*1 ; maskAtl[...] = True
idxa = np.argwhere(maskg == 1).transpose()
maskAtl[idxa[0],idxa[1]] = False
maskPac = maski*1 ; maskPac[...] = True
idxp = np.argwhere(maskg == 2).transpose()
maskPac[idxp[0],idxp[1]] = False
maskInd = maski*1 ; maskInd[...] = True
idxi = np.argwhere(maskg == 3).transpose()
maskInd[idxi[0],idxi[1]] = False

# Compute total volume as thickness*area of each cell
volEN4 = thick*(1-(thick.mask).astype(int))*areaisig
print(np.ma.sum(volEN4)/1.e09) # In km3

# Mask with basin masks
thicki = thick*1.
thicki.mask = maski
thickia = thick*1.
thickia.mask = maskAtl
thickip = thick*1.
thickip.mask = maskPac
thickii = thick*1.
thickii.mask = maskInd
# Compute zonal volume
volz  = np.ma.sum(thicki *(1- (thicki.mask).astype(int))*areaisig, axis=2)
volza = np.ma.sum(thickia*(1- (thickia.mask).astype(int))*areaisig, axis=2)
volzp = np.ma.sum(thickip*(1- (thickip.mask).astype(int))*areaisig, axis=2)
volzi = np.ma.sum(thickii*(1- (thickii.mask).astype(int))*areaisig, axis=2)
print(np.ma.sum(volz)/1.e09)

isonvol = np.ma.stack([volz,volza,volzp,volzi],axis=0)
isonvol[isonvol==0]=np.ma.masked

# -- Read variables
lat = f.variables['latitude'][:]
density = f.variables['lev'][:]

basinN=4
latN = len(lat)
densityN = len(density)

# -- Z grid for calculating ocean volume per depth level
gridz2 = np.concatenate([np.arange(0,21,1),np.arange(25,5501,5)])

# WOA13 grid
targetz = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
   85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375,
   400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
   1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550,
   1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2300,
   2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500,
   3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700,
   4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500]

# -- Read bathymetry
# Read masks
fmask = open_ncfile('/home/ysilvy/Density_bining/Yona_analysis/data/170224_WOD13_masks.nc','r')
basinmask = fmask.variables['basinmask3'][:] # (latitude, longitude)
depthmask = fmask.variables['depthmask'][:] # (latitude, longitude)
longitude = fmask.variables['longitude'][:]
# Create basin masks
mask_a = basinmask != 1
mask_p = basinmask != 2
mask_i = basinmask != 3
# Read bathy
depthmask_a = np.ma.array(depthmask, mask=mask_a) # Mask every basin except Atlantic
depthmask_p = np.ma.array(depthmask, mask=mask_p)
depthmask_i = np.ma.array(depthmask, mask=mask_i)
# Zonal bathy (for plotting afterwards)
bathy_a = np.ma.max(depthmask_a, axis=1)
bathy_p = np.ma.max(depthmask_p, axis=1)
bathy_i = np.ma.max(depthmask_i, axis=1)
# Read area
area = fmask.variables['basinmask3_area'][:] # (latitude,longitude)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# ===
# === CALCULATE THEORETICAL OCEAN VOLUME PER DEPTH LEVEL FROM BATHYMETRY ===
# ===

# Compute zonal volume on gridz2 levels (every 5 meters, every 1 meter at the surface)
V = np.ma.masked_all((basinN,len(gridz2),latN))
# Loop on Atlantic, Pacific and Indian basins
start_time = time.time()
for ibasin in range(1,4): 
    print(ibasin)
    if ibasin==1:
        depthmask = depthmask_a
    elif ibasin==2:
        depthmask = depthmask_p
    else:
        depthmask = depthmask_i
    for ilat in range(latN):
        area_lat = np.ma.average(area[ilat,:]) # Same area for all longitudes along ilat: take the average to make sure we don't choose a masked gridpoint
        # Loop on depths
        for iz in range(len(gridz2)):
            cells_above = np.argwhere(depthmask[ilat,:]>gridz2[iz]) #.squeeze() # list of indices where bathymetry is below the current z level
            if len(cells_above)!=0: #Exclude depths where there is no water
                S = len(cells_above)*area_lat # total area in km2 at depth gridz[iz] for the current band of latitude
                V[ibasin,iz,ilat] = S*0.001*(gridz2[iz+1]-gridz2[iz]) # Corresponding volume in km3 (5m interval for gridz, 1m at surface)


# Now compute volume for global zonal mean (ibasin=0)
V[0,:,:] = np.sum(V[1:,:,:],axis=0)

# ========
# PSEUDO-Z CONSTRUCTION
# ========

# Constructing remapping relationship pseudo_depth[basin,density,latitude] by re-filling the ocean from the surface down with horizontal layers of constant density

# Initialize
pseudo_depth = np.ma.masked_all((basinN,densityN,latN))
# Start loops
for ibasin in range(basinN):
    if ibasin==1:
        depthmask = depthmask_a
    elif ibasin==2:
        depthmask = depthmask_p
    elif ibasin==3:
        depthmask = depthmask_i
    for ilat in range(latN):
        if not np.ma.is_masked(np.all(isonvol[ibasin,:,ilat])):
            idx_range = np.ma.flatnotmasked_edges(isonvol[ibasin,:,ilat]) # Indices edges of unmasked densities in the column
            if ibasin==0: # For the whole ocean, take max depth
                bathy = np.ma.max([np.ma.max(depthmask_a[ilat,:]),np.ma.max(depthmask_p[ilat,:]),np.ma.max(depthmask_i[ilat,:])])
            else:
                bathy=np.ma.max(depthmask[ilat,:]) # Find max depth of the water column
            ibathy=np.argwhere(gridz2==bathy).squeeze() # Index of bathymetry = first masked value of the water column
            cum_V = np.cumsum(V[ibasin,:,ilat])
            cum_isonvol = np.cumsum(isonvol[ibasin,:,ilat])
            # Loop on density levels (stop one before the end)
            for ilev in range(idx_range[0],idx_range[1]):
                isonvol_lev = isonvol[ibasin,ilev,ilat]
                iz = find_nearest(cum_V,cum_isonvol[ilev])# Find index where the cumulated volume at ilev is closest to cumulated volume on gridz2
                pseudo_depth[ibasin,ilev,ilat] = gridz2[iz] # Save corresponding z level on gridz
            # Last density level: corresponds to bathy
            pseudo_depth[ibasin,idx_range[1],ilat] = bathy 
            
# ========
# SAVE PSEUDO-Z TO FILE
# ========

import pickle

# write to pickle
pickle.dump( pseudo_depth, open( "/home/ysilvy/Density_bining/Yona_analysis/data/remaptoz/EN4.pseudo_depth.zonal.pkl", "wb" ), protocol=2 )

