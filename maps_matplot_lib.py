#!/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import numpy.ma as ma
#from mpl_toolkits.basemap import Basemap
#from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset as open_ncfile
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

# --------------------------------
#       Variable properties
# --------------------------------

def defVar(longName):
    salinity = {
        'longN': 'salinity',  # long name
        'var': 'sog',  # variable name
        'minmax': [-0.3, 0.3, 16],  # for diff shading + number of color interval
        'clevsm': np.arange(30, 40, .25),  # for mean contours
        'clevsm_bold': np.arange(30, 40, 0.5),
        'clevsmdif': np.arange(-.2, .2, .025),  # for mean contours
        'clevsmstd': np.arange(0., .2, .005),  # for stddev contours
        '1dminmax': [-.1, .1], # for 1D ToE plots
        'legVar': "Salinity",  # Legend name
        'unit': "PSS-78",  # TODO: could be read from file
    }

    temp = {'var': 'thetaog', 'minmax': [-0.65, 0.65, 14], 'clevsm': np.arange(-2, 30, 1), 'clevsm_bold': np.arange(-2,30,2),
            'clevsmstd': np.arange(0, 2., .01), '1dminmax': [-.4, .4],'clevsmdif': np.arange(-.4, .4, .05),
            'legVar': "Temperature", 'unit': "C", 'longN': 'temp',
            }
    depth = {'var': 'isondepthg', 'minmax': [-75., 75., 10], 'clevsm': np.arange(0, 2000, 100),
             'clevsmstd': np.arange(0, 20, 5),'1dminmax': [-10, 50],'clevsmdif': np.arange(-75, 75, 10),
             'legVar': "Depth", 'unit': "m", 'longN': 'depth',
             }
    volume = {'var': 'isonvol', 'minmax': [-20., 20., 20], 'clevsm': np.arange(0, 200, 20),
              'clevsmstd': np.arange(0, 20, 1),'1dminmax': [-5, 5],'clevsmdif': np.arange(-20, 20, 5),
              'legVar': "Volume", 'unit': "1.e12 m^3", 'longN': 'volume',
              }
    persist = {'var': 'isonpers', 'minmax': [-10., 10., 20], 'clevsm': np.arange(0, 90, 10),
               'clevsmstd': np.arange(0, 3., .5),'1dminmax': [-50, 50],'clevsmdif': np.arange(-10, 10, 2),
               'legVar': "Persistence", 'unit': "% of time", 'longN': 'persist'
               }
    heatcontent = {'var': 'isonhtc', 'minmax': [-10., 10., 20], 'clevsm': np.arange(0, 90, 10),
               'clevsmstd': np.arange(0, 3., .5),'1dminmax': [-50, 50],'clevsmdif': np.arange(-10, 10, 2),
               'legVar': "Heat content", 'unit': "10^XX J", 'longN': 'heatcontent'
               }

    vars = [salinity, temp, depth, volume, persist, heatcontent]

    varout = 'None'
    for ivar in range(len(vars)):
        if vars[ivar]['longN'] == longName:
            varout = vars[ivar]

    return varout


def defVarDurack(longName):

    degree_sign= u'\N{DEGREE SIGN}'

    salinity = {'var_change': 'salinity_change', 'var_change_er':'thetao_change_error',
                'var_mean': 'salinity_mean',
                'var_mean_zonal': 'salinity_mean_basin_zonal',
                'var_change_zonal': 'salinity_change_basin_zonal', 'var_change_zonal_er' : 'salinity_change_error_basin_zonal',
                'minmax': [-0.3, 0.3, 16],
                'minmax_zonal': [-0.2, 0.2, 16],
                'clevsm': np.arange(30, 40, .25),
                'clevsm_zonal': np.arange(30, 40, .25), #0.1
                'clevsm_bold': np.arange(30, 40, .5),
                'legVar': "Salinity", 'unit': "PSS-78", 'longN': 'salinity'}

    temp = {'var_change':'thetao_change', 'var_change_er':'thetao_change_error',
            'var_mean':'thetao_mean',
            'var_mean_zonal': 'thetao_mean_basin_zonal',
            'var_change_zonal': 'thetao_change_basin_zonal', 'var_change_zonal_er' : 'thetao_change_error_basin_zonal',
            'minmax': [-0.65, 0.65, 14],
            'minmax_zonal' : [-0.5,0.5,16],
            'clevsm': np.arange(-2, 30, 1),
            'clevsm_zonal': np.arange(-2, 30, 1),
            'clevsm_bold': np.arange(-2,30,2),
            'legVar': "Temperature", 'unit': degree_sign+"C", 'longN': 'temp'}

    vars = [salinity,temp]
    for ivar in range(len(vars)):
        if vars[ivar]['longN'] == longName:
            varout = vars[ivar]

    return varout


def defVarmme(longName):

    degree_sign= u'\N{DEGREE SIGN}'

    salinity = {'var_zonal': 'isonsoBowl', 'var_zonal_w/bowl': 'isonso',
                'var_global': 'sogBowl', 'var_global_std':'sogBowlStd',
                'minmax': [-0.3, 0.3, 16],
                'minmax_zonal': [-0.2, 0.2, 16], 'minmax_zonal_rcp85': [-0.6,0.6,16],
                'clevsm': np.arange(30, 40, .25),
                'clevsm_zonal': np.arange(30, 40, .1),
                'clevsm_bold': np.arange(30, 40, .5),
                'legVar': "Salinity", 'unit': "PSS-78", 'longN': 'salinity'}

    temp = {'var_zonal':'isonthetaoBowl', 'var_zonal_w/bowl': 'isonthetao',
            'var_global': 'thetaogBowl', 'var_global_std':'thetaogBowlStd',
            'minmax': [-0.65, 0.65, 14],
            'minmax_zonal' : [-0.5,0.5,16], 'minmax_zonal_rcp85': [-1,1,16],
            'clevsm': np.arange(-2, 30, 1),
            'clevsm_zonal': np.arange(-2, 30, 1),
            'clevsm_bold': np.arange(-2,30,2),
            'legVar': "Temperature", 'unit': degree_sign+"C", 'longN': 'temp'}

    depth = {'var_zonal':'isondepthBowl', 'var_zonal_w/bowl': 'isondepth',
             'clevsm_zonal': np.arange(0, 2000, 100),
             'clevsm_bold' : np.arange(0,2000,500),
             'minmax_zonal': [-50, 50, 16], 'minmax_zonal_rcp85': [-300, 300, 16],
             'legVar': "Depth", 'unit': "m", 'longN': 'depth'}

    volume = {'var_zonal': 'isonvolBowl', 'var_zonal_w/bowl': 'isonvol',
              'minmax_zonal': [-20., 20., 16], 'minmax_zonal_rcp85': [-40,40,16],
              'clevsm_zonal': np.arange(0, 500, 50),
              'legVar': "Volume", 'unit': "1.e12 m^3", 'longN': 'volume'
              }

    vars = [salinity,temp,depth,volume]
    for ivar in range(len(vars)):
        if vars[ivar]['longN'] == longName:
            varout = vars[ivar]

    return varout


# Customed colormap
def custom_div_cmap(numcolors=17, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 17 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors=[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


# ----------------------------------------------------
#   Build zonal latitude/density plot
# ----------------------------------------------------

def zonal_2D(plt, action, ax0, ax1, ticks, lat, density, varBasin, cnDict, domrho, clevsm=None, clevsm_bold=None):

    # latitude domain
    domlat = [-70, 70]

    if action == 'total' :
        var = varBasin['var_change']
        var_mean = varBasin['var_mean']
        var_er = varBasin['var_error']

        # -- Error field for Durack & Wijffels data
        var_er = var_er * 1.1  # to account for a potential underestimation of the error determined by a bootstrap analysis
        var_er = var_er * 1.64  # 90% confidence level
        not_signif_change = np.where(np.absolute(var) < var_er, 1, 0)

        # -- Format for contour labels
        levfmt = '%.0f'
        if abs(clevsm[1] - clevsm[0]) < 1:
            levfmt = '%.1f'
        if abs(clevsm[1] - clevsm[0]) < 0.1:
            levfmt = '%.2f'

    elif action == 'total_mme':
        var = varBasin['var_change']
        var_mean = varBasin['var_mean']

        if np.any(clevsm) != None:
            # -- Format for contour labels
            levfmt = '%.0f'
            if abs(clevsm[1] - clevsm[0]) < 1:
                levfmt = '%.1f'
            if abs(clevsm[1] - clevsm[0]) < 0.1:
                levfmt = '%.2f'

    else :
        var = varBasin[action]

    if action != 'total' and action != 'total_mme' and action != 'ToE':
        bowl = varBasin['bowl']
    if (action == 'total_mme' or action == 'ToE') and varBasin['labBowl'] != None :
        bowl2 = varBasin['bowl2']
        bowl1 = varBasin['bowl1']
        label1 = varBasin['labBowl'][0]
        label2 = varBasin['labBowl'][1]

    # levels and color map
    levels = cnDict['levels']
    cmap = cnDict['cmap']

    # Create meshgrid
    lat2d, density2d = np.meshgrid(lat, density)

    # ==== Upper panel ====

    cnplot1 = ax0.contourf(lat2d, density2d, var, levels=levels, cmap=cmap, extend=cnDict['ext_cmap'])

    # -- Draw areas where signal is not significant for D&W
    if action == 'total' :
        error_plot = ax0.contourf(lat2d, density2d, not_signif_change, levels=[0.25,0.5,1.5], colors='None',
                                   hatches=['','....'])

    if (action == 'ToE' or action == 'total_mme') and varBasin['labBowl'] != None:
        ax0.plot(lat, bowl2, linewidth=2, color='black',label=label2)
        ax0.plot(lat, bowl1, linestyle = '--', linewidth=2, color='black',label=label1)
        # -- Add legend for bowl position
        if varBasin['name'] == 'Indian':
            ax0.legend(loc='upper right', title='Bowl', fontsize=12)

    ax0.set_ylim([domrho[0], domrho[1]])
    ax0.set_xlim([domlat[0], domlat[1]])
    ax0.invert_yaxis()
    ax0.tick_params(
        axis='x',  # changes apply to the x axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False,
        top=False)
    
    ax0.yaxis.set_tick_params(which='major',width=2)

    ax0.tick_params(axis='y',right=True)
    if ticks != 'left':
        ax0.tick_params(axis='y', labelleft=False)
    if ticks == 'right':
        ax0.tick_params(axis='y', labelright=True)

    plt.setp(ax0.get_yticklabels(), fontweight='bold', fontsize=12)
    ax0.axvline(x=0, color='black', ls='--')


    # === Lower panel ====

    cnplot2 = ax1.contourf(lat2d, density2d, var, levels=levels, cmap=cmap, extend=cnDict['ext_cmap'])

    # -- Draw areas where signal is not significant for D&W
    if action == 'total' :
        error_plot = ax1.contourf(lat2d, density2d, not_signif_change, levels=[0.25, 0.5, 1.5], colors='None',
                                  hatches=['', '....'])

    if (action == 'ToE' or action == 'total_mme') and varBasin['labBowl'] != None:
        ax1.plot(lat, bowl2, linewidth=2, color='black',label=label2)
        ax1.plot(lat, bowl1, linestyle = '--', linewidth=2, color='black',label=label1)


    ax1.set_ylim([domrho[1], domrho[2]])
    ax1.set_xlim([domlat[0], domlat[1]])
    ax1.invert_yaxis()
    ax1.tick_params(
        axis='x',  # changes apply to the x axis
        which='both',  # both major and minor ticks are affected
        top=False)  # ticks along the top edge are off
    
    ax1.tick_params(axis='y',right=True)
    if ticks != 'left':
        ax1.tick_params(axis='y', labelleft=False)
    if ticks == 'right':
        ax1.tick_params(axis='y', labelright=True)

    plt.setp(ax1.get_yticklabels(), fontweight='bold', fontsize=12)
    plt.setp(ax1.get_xticklabels(), fontweight='bold', fontsize=12)
    ax1.yaxis.set_tick_params(which='major',width=2)
    ax1.xaxis.set_tick_params(which='major',width=2)
    
    ax1.axvline(x=0, color='black', ls='--')

    # Re-label x-axis
    xlabels = ['', '60S', '40S', '20S', '0', '20N', '40N', '60N']
    ax1.set_xticklabels(xlabels)

    # Remove intersecting tick at rhomid
    yticks = ax1.yaxis.get_major_ticks()
    if ticks == 'left':
        yticks[0].label1.set_visible(False)
    if ticks == 'right':
        yticks[0].label2.set_visible(False)


    # -- add plot title
    ax0.text(-60, 22, varBasin['name'], fontsize=14, fontweight='bold')

    cnplot = [cnplot1, cnplot2]

    return cnplot


def modelagree(ax0,ax1,agreelev,lat,lev,var_agree):

    # Create meshgrid
    lat2d, lev2d = np.meshgrid(lat, lev)

    # -- draw agreement contour > agreement level (agreelev)
    ax0.contourf(lat2d, lev2d, var_agree, levels=[-agreelev, agreelev], hatches=['....'], colors='None')
    #ax0.contour(lat2d, lev2d, var_agree, [agreelev - .0001, agreelev + 0.00001], colors='0.3',
    #            linewidths=1.5)
    #ax0.contour(lat2d, lev2d, var_agree, [-agreelev - .0001, -agreelev + 0.00001], colors='0.3',
    #            linewidths=1.5)

    ax1.contourf(lat2d, lev2d, var_agree, levels=[-agreelev, agreelev], hatches=['....'], colors='None')
    #ax1.contour(lat2d, lev2d, var_agree, [agreelev - .0001, agreelev + 0.00001], colors='0.3',
    #            linewidths=1.5)
    #agree_plot = ax1.contour(lat2d, lev2d, var_agree, [-agreelev - .0001, -agreelev + 0.00001], colors='0.3',
    #                         linewidths=1.5)

    #return agree_plot


# -----------------------------------------------
#          Average in lat/rho domain
# -----------------------------------------------

def averageDom(field, dim, domain, lat, rho):
    
    latidx = np.argwhere((lat >= domain[0]) & (lat <= domain[1])).transpose()
    rhoidx = np.argwhere((rho >= domain[2]) & (rho <= domain[3])).transpose()
    lidx1 = latidx[0][0];
    lidx2 = latidx[0][-1]
    ridx1 = rhoidx[0][0];
    ridx2 = rhoidx[0][-1]
    if dim == 3:
        vara = np.ma.average(field[:, ridx1:ridx2, :], axis=1)
        var_ave = np.ma.average(vara[:, lidx1:lidx2], axis=1)
    else:
        vara = np.ma.average(field[ridx1:ridx2, :], axis=0)
        var_ave = np.ma.average(vara[lidx1:lidx2], axis=0)

    return var_ave


# ----------------------------------------------------
#   Build zonal latitude/depth plot
# ----------------------------------------------------

def zon_2Dz(plt, ax0, ax1, ticks, lat, lev, varBasin, cnDict, domzed, clevsm=None, clevsm_bold=None):

    # -- variables
    var = varBasin['var_change']
    
    # -- min,mid,max depth for the 2 panels
    zedmin = domzed[0]
    zedmid = domzed[1]
    zedmax = domzed[2]

    # -- Latmin/max
    domlat = [-70, 70]

    # Create meshgrid
    lat2d, lev2d = np.meshgrid(lat, lev)

    #
    # ====  Upper panel  ===================================================
    #
    ax0.axis([domlat[0], domlat[1], zedmin, zedmid])
    ax0.invert_yaxis()
    ax0.tick_params(
        axis='x',  # changes apply to the x axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        labelbottom=False,
        top=False)
    ax0.tick_params(axis='y',which='both',right=True)

    if ticks != 'left':
        ax0.tick_params(axis='y', labelleft=False)
    if ticks == 'right':
        ax0.tick_params(axis='y', labelright=True)
    
    plt.setp(ax0.get_yticklabels(), fontweight='bold', fontsize=12)
    
    ax0.axvline(x=0, color='black', ls='--')

    # -- draw filled contours of the field
    cnplot0 = ax0.contourf(lat2d, lev2d, var, cmap=cnDict['cmap'], levels=cnDict['levels'], extend=cnDict['ext_cmap'])

    if clevsm != None:
        # -- Format for contour labels
        levfmt='%.0f'
        if abs(clevsm[1]-clevsm[0]) < 1:
            levfmt='%.1f'
        if abs(clevsm[1]-clevsm[0]) < 0.1:
            levfmt='%.2f'

        # -- draw mean field contours
        cpplot11 = ax0.contour(lat2d, lev2d, varBasin['var_mean'], clevsm, colors='black', linewidths=0.5)
        cpplot12 = ax0.contour(lat2d, lev2d, varBasin['var_mean'], clevsm_bold, colors='black', linewidths=2)
        ax0.clabel(cpplot12, inline=1, fontsize=12, fontweight='bold', fmt=levfmt)

    # -- Draw isopycnals
    if cnDict['isopyc'] == True:
        levels1 = np.arange(21,28.6,0.5)
        levels2 = np.arange(21,28.6,1)
        ax0.contour(lat2d, lev2d, varBasin['density'], levels=levels1, colors='black', linewidths=0.5)
        cont_isopyc1 = ax0.contour(lat2d, lev2d, varBasin['density'], levels=levels2, colors='black', linewidths=2)
        ax0.clabel(cont_isopyc1, inline=1, fontsize=13, fmt='%d')
        
    ax0.yaxis.set_tick_params(which='major',width=2)
    #
    # ====  Lower panel   ===================================================
    #
    ax1.axis([domlat[0], domlat[1], zedmid, zedmax])
    ax1.invert_yaxis()
    ax1.tick_params(
        axis='x',  # changes apply to the x axis
        which='both',  # both major and minor ticks are affected
        top=False)  # ticks along the bottom edge are off
    ax1.tick_params(axis='y',which='both',right=True)
    
    if ticks != 'left':
        ax1.tick_params(axis='y', labelleft=False)
    if ticks == 'right':
        ax1.tick_params(axis='y', labelright=True)

    ax1.axvline(x=0, color='black', ls='--')

    # -- Re-label x-axis
    xlabels = ['', '60S', '40S', '20S', '0', '20N', '40N', '60N']
    ax1.set_xticklabels(xlabels,fontweight='bold',fontsize=12)
    # -- Set y ticks
    if zedmax == 2000:
        ymajorLocator = MultipleLocator(500)
        yminorLocator = AutoMinorLocator(5)
    else:
        ymajorLocator = MultipleLocator(1000)
        yminorLocator = AutoMinorLocator(2)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
    plt.setp(ax1.get_yticklabels(), fontweight='bold', fontsize=12)
        
    # -- draw filled contours
    cnplot1 = ax1.contourf(lat2d, lev2d, var, cmap=cnDict['cmap'], levels=cnDict['levels2'], extend=cnDict['ext_cmap'])

    if clevsm != None:
        # -- draw mean field contours
        cpplot21 = ax1.contour(lat2d, lev2d, varBasin['var_mean'], clevsm, colors='black', linewidths=0.5)
        cpplot22 = ax1.contour(lat2d, lev2d, varBasin['var_mean'], clevsm_bold, colors='black', linewidths=2)
        ax1.clabel(cpplot22, inline=1, fontsize=12, fontweight='bold', fmt=levfmt)

    # -- Draw isopycnals
    if cnDict['isopyc'] == True:
        ax1.contour(lat2d, lev2d, varBasin['density'], levels=levels1, colors='black', linewidths=0.5)
        cont_isopyc2 = ax1.contour(lat2d, lev2d, varBasin['density'], levels=levels2, colors='black', linewidths=2)
        ax1.clabel(cont_isopyc2, inline=1, fontsize=13, fmt='%d')

    ax1.yaxis.set_tick_params(which='major',width=2)
    ax1.xaxis.set_tick_params(which='major',width=2)
        
    # -- add plot title
    ax1.text(domlat[1] - 37, zedmax-300, varBasin['name'], fontsize=15, fontweight='bold',bbox=dict(facecolor='white',edgecolor='white'))


    cnplot = [cnplot0, cnplot1]
    return cnplot
