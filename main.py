# -*- coding: utf-8 -*-
"""
2D Compressible Navier-Stokes solver

Boundary condition options:
    -'wall' for bc_type will imply no-slip, dp and drho of 0; T must be specified
    -'zero_grad' will impose 0 normal gradient of that variable
    
Features to include (across all classes):
    -CoolProp library for material properties (track down needed functions)
    -speed of sound calculation (SolverClasses?)
    -Fix biasing meshing tools (this script and GeomClasses)
        ->Figure out biasing wrt dx and dy array sizes and mesh griding those (GeomClasses)
    -File reader for settings
    -periodic boundary conditions (SolverClasses)

@author: Joseph
"""

##########################################################################
# ----------------------------------Libraries and classes
##########################################################################
#import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import os
import CoolProp.CoolProp as CP

#from GeomClasses import OneDimLine as OneDimLine
from GeomClasses import TwoDimPlanar as TwoDimPlanar
#import MatClasses as Mat
import SolverClasses as Solvers
import FileClasses

##########################################################################
# ------------------------------ Geometry, Domain and BCs Setup
#    Reference directions:
#    left-smallest x coordinate
#    right-largest x value
#    north-largest y coordinate
#    south-smallest y coordinate
##########################################################################
settings={} # Dictionary of problem settings
BCs={} # Dictionary of boundary conditions
# Geometry details
settings['Length']                  = 4.0
settings['Width']                   = 1.0
settings['Nodes_x']                 = 51
settings['Nodes_y']                 = 101
settings['Fluid']                   = 'Air'
settings['k']                       = CP.PropsSI('L','T', 300, 'P', 101325, settings['Fluid']) # If using constant value
settings['gamma']                   = CP.PropsSI('Cpmass','T',300,'P',101325,settings['Fluid'])/CP.PropsSI('Cvmass','T',300,'P',101325,settings['Fluid'])
settings['R']                       = CP.PropsSI('gas_constant','Air')/CP.PropsSI('M',settings['Fluid']) # Specific to fluid
settings['mu']                      = CP.PropsSI('V','T', 300, 'P', 101325, settings['Fluid'])

# Meshing details
settings['bias_type_x']             = None
settings['bias_size_x']             = 0.003 # Smallest element size (IN PROGRESS)
settings['bias_type_y']             = None
settings['bias_size_y']             = 10**(-6) # Smallest element size (IN PROGRESS)

# Boundary conditions
BCs['bc_type_left']                 = 'periodic'
BCs['bc_left_rho']                  = None
BCs['bc_left_u']                    = None
BCs['bc_left_v']                    = None
BCs['bc_left_p']                    = None
BCs['bc_left_T']                    = 'zero_grad'
BCs['bc_type_right']                = 'periodic'
BCs['bc_right_rho']                 = None
BCs['bc_right_u']                   = None
BCs['bc_right_v']                   = None
BCs['bc_right_p']                   = None
BCs['bc_right_T']                   = 'zero_grad'
BCs['bc_type_south']                = 'wall'
BCs['bc_south_rho']                 = None
BCs['bc_south_u']                   = None
BCs['bc_south_v']                   = None
BCs['bc_south_p']                   = None
BCs['bc_south_T']                   = 1000
BCs['bc_type_north']                = 'wall'
BCs['bc_north_rho']                 = None
BCs['bc_north_u']                   = None
BCs['bc_north_v']                   = None
BCs['bc_north_p']                   = None
BCs['bc_north_T']                   = 300

# Initial conditions ????


# Time advancement
settings['CFL']                     = 0.05
settings['total_time_steps']        = 30
settings['Time_Scheme']             = 'Exp'


print 'Initializing geometry package...'
#domain=OneDimLine(L,Nx)
domain=TwoDimPlanar(settings)
domain.mesh()

##########################################################################
# -------------------------------------Initialize solver and domain
##########################################################################

print 'Initializing solver package...'
solver=Solvers.TwoDimPlanarSolve(domain, settings, BCs)

print 'Initializing domain...'
domain.rho[:,:]=CP.PropsSI('D','T',300,'P',101325,settings['Fluid'])
domain.u[:,:]=0
domain.v[:,:]=0
domain.T[:,:]=300
#domain.p[:,:]=101325

domain.p[:,:]=domain.rho[:,:]*domain.R*domain.T[:,:]
solver.Apply_BCs(domain.rho, domain.rhou, domain.rhov, \
                 domain.rhoE, domain.u, domain.v, domain.p, domain.T)
#domain.T[1:-1,1:-1]=domain.p[1:-1,1:-1]/domain.rho[1:-1,1:-1]/domain.R

domain.rhou[:,:]=domain.rho[:,:]*domain.u[:,:]
domain.rhov[:,:]=domain.rho[:,:]*domain.v[:,:]
domain.rhoE[:,:]=domain.rho[:,:]*(0.5*(domain.u[:,:]**2+domain.v[:,:]**2) \
           +domain.Cv*domain.T[:,:])

##########################################################################
# -------------------------------------File setups
##########################################################################
print 'Initializing files...'
os.chdir('Tests')
datTime=str(datetime.date(datetime.now()))+'_'+'{:%H%M}'.format(datetime.time(datetime.now()))
isBinFile=False

#output_file=FileClasses.FileOut('Output_'+datTime, isBinFile)
input_file=FileClasses.FileOut('Input_'+datTime, isBinFile)

# Write headers to files
input_file.header('INPUT')
#output_file.header('OUTPUT')

# Write input file with settings
input_file.input_writer(settings, BCs)
input_file.close()

##########################################################################
# -------------------------------------Solve
##########################################################################
print('######################################################')
print('#              2D Navier-Stokes Solver               #')
print('#              Created by J. Mark Epps               #')
print('#          Part of Masters Thesis at UW 2018-2020    #')
print('######################################################\n')

print 'Solving:'
for nt in range(settings['total_time_steps']):
    print 'Time step %i of %i'%(nt+1, settings['total_time_steps'])
    err=solver.Advance_Soln()
    if err==1:
        print '#################### Solver aborted #######################'
        break

#output_file.close()

##########################################################################
# ------------------------------------Plots
##########################################################################
# 2D plot
#fig=pyplot.figure(figsize=(7, 7), dpi=100)
#ax = fig.gca(projection='3d')
#ax.plot_surface(domain.X, domain.Y, domain.T, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
##ax.set_xlim(0,0.001)
##ax.set_ylim(0.005,0.006)
#ax.set_zlim(300, 1000)
#ax.set_xlabel('$x$ (m)')
#ax.set_ylabel('$y$ (m)')
#ax.set_zlabel('T (K)')
#
## 1D Plot
##fig2=pyplot.figure(figsize=(7,7))
##pyplot.plot(domain.Y[:,1]*1000, domain.T[:,1],marker='x')
##pyplot.xlabel('$y$ (mm)')
##pyplot.ylabel('T (K)')
##pyplot.title('Temperature distribution at 2nd x')
##pyplot.xlim(5,6)
#
## Velocity Quiver plot and pressure contour
#pl=2
#fig3=pyplot.figure(figsize=(7, 7), dpi=100)
#pyplot.quiver(domain.X[::pl, ::pl], domain.Y[::pl, ::pl], \
#              domain.u[::pl, ::pl], domain.v[::pl, ::pl]) 
#pyplot.contourf(domain.X, domain.Y, domain.p, alpha=0.5, cmap=cm.viridis)  
#pyplot.colorbar()
#pyplot.xlabel('$x$ (m)')
#pyplot.ylabel('$y$ (m)')

print('Solver has finished its run')