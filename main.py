# -*- coding: utf-8 -*-
"""
2D Compressible Navier-Stokes solver

Boundary condition options:
    -'wall' for bc_type will imply no-slip, dp of 0; T must be specified
    -'zero_grad' will impose 0 normal gradient of that variable
    
Features to include (across all classes):
    -CoolProp library for material properties (track down needed functions)
    -Fix biasing meshing tools (this script and GeomClasses)
        ->Figure out biasing wrt dx and dy array sizes and mesh griding those (GeomClasses)
    -File reader for settings
    -periodic boundary conditions (SolverClasses)

ME765 project setup:
BCs['bc_type_left']                 = 'periodic'
BCs['bc_left_rho']                  = None
BCs['bc_left_u']                    = None
BCs['bc_left_v']                    = None
BCs['bc_left_p']                    = None
BCs['bc_left_T']                    = None
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_right']                = 'periodic'
BCs['bc_right_rho']                 = None
BCs['bc_right_u']                   = None
BCs['bc_right_v']                   = None
BCs['bc_right_p']                   = None
BCs['bc_right_T']                   = None
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_south']                = 'wall'
BCs['bc_south_rho']                 = None
BCs['bc_south_u']                   = None
BCs['bc_south_v']                   = None
BCs['bc_south_p']                   = None
BCs['bc_south_T']                   = 600
# numpy.linspace(400, 900, settings['Nodes_x'])
BCs['bc_type_north']                = 'wall'
BCs['bc_north_rho']                 = None
BCs['bc_north_u']                   = None
BCs['bc_north_v']                   = None
BCs['bc_north_p']                   = None
BCs['bc_north_T']                   = 300

Inlet/outlet
BCs['bc_type_left']                 = 'inlet'
BCs['bc_left_rho']                  = None
BCs['bc_left_u']                    = 1.0
BCs['bc_left_v']                    = 0
BCs['bc_left_p']                    = 2*101325
BCs['bc_left_T']                    = 300
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_right']                = 'outlet'
BCs['bc_right_rho']                 = None
BCs['bc_right_u']                   = None
BCs['bc_right_v']                   = None
BCs['bc_right_p']                   = 101325
BCs['bc_right_T']                   = None
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_south']                = 'wall'
BCs['bc_south_rho']                 = None
BCs['bc_south_u']                   = None
BCs['bc_south_v']                   = None
BCs['bc_south_p']                   = None
BCs['bc_south_T']                   = 600
# numpy.linspace(400, 900, settings['Nodes_x'])
BCs['bc_type_north']                = 'wall'
BCs['bc_north_rho']                 = None
BCs['bc_north_u']                   = None
BCs['bc_north_v']                   = None
BCs['bc_north_p']                   = None
BCs['bc_north_T']                   = 600
# numpy.linspace(400, 900, settings['Nodes_x'])

@author: Joseph
"""

##########################################################################
# ----------------------------------Libraries and classes
##########################################################################
import numpy
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
settings['Width']                   = 2.0
settings['Nodes_x']                 = 51
settings['Nodes_y']                 = 101
settings['Fluid']                   = 'Air'
#CP.PropsSI('L','T', 300, 'P', 101325, settings['Fluid'])
settings['k']                       = CP.PropsSI('L','T', 300, 'P', 101325, settings['Fluid']) # If using constant value
settings['gamma']                   = CP.PropsSI('Cpmass','T',300,'P',101325,settings['Fluid'])/CP.PropsSI('Cvmass','T',300,'P',101325,settings['Fluid'])
settings['R']                       = CP.PropsSI('gas_constant','Air')/CP.PropsSI('M',settings['Fluid']) # Specific to fluid
settings['mu']                      = CP.PropsSI('V','T', 300, 'P', 101325, settings['Fluid'])
settings['Gravity_x']               = 0
settings['Gravity_y']               = 0

# Meshing details
"""
Biasing options:
    -'OneWayUp'   for linearly increasing element sizes with increasing x/y
    -'OneWayDown' for linearly decreasing element sizes with increasing x/y
    -'TwoWayEnd'  for linearly increasing sizes till middle, then decrease again
    -'TwoWayMid'  for linearly decreasing sizes till middle, then increase again
    -size         is the smallest element size based on above selection
"""
settings['bias_type_x']             = None
settings['bias_size_x']             = 0.005 # Smallest element size
settings['bias_type_y']             = None
settings['bias_size_y']             = 0.0005 # Smallest element size

# Boundary conditions
"""
Options:
    -'periodic': no properties need to be specified; implied on opposite face too
    -'periodic': [IN PROGRESS] specify pressure, is poiseuille flow
    -'wall': specify T; no slip and dp=0 enforced implicitly
    -'wall': T gradient as ('grad',[value]); no slip and dp=0 enforced implicitly
    -'outlet': specify pressure; rest is calculated from interior points
    -'inlet': specify velocities, temperature and pressure
Profiles possible; must be same size as number of nodes on that boundary
'zero_grad' assumes 0 gradient of that variable (walls only so far)
('grad',[value]) enforces a gradient normal to boundary (temperature on walls)
"""
BCs['bc_type_left']                 = 'periodic'
BCs['bc_left_rho']                  = None
BCs['bc_left_u']                    = None
BCs['bc_left_v']                    = None
BCs['bc_left_p']                    = None
BCs['bc_left_T']                    = None
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_right']                = 'periodic'
BCs['bc_right_rho']                 = None
BCs['bc_right_u']                   = None
BCs['bc_right_v']                   = None
BCs['bc_right_p']                   = None
BCs['bc_right_T']                   = None
# numpy.linspace(400, 900, settings['Nodes_y'])
BCs['bc_type_south']                = 'wall'
BCs['bc_south_rho']                 = None
BCs['bc_south_u']                   = None
BCs['bc_south_v']                   = None
BCs['bc_south_p']                   = None
BCs['bc_south_T']                   = 600
# numpy.linspace(400, 900, settings['Nodes_x'])
BCs['bc_type_north']                = 'wall'
BCs['bc_north_rho']                 = None
BCs['bc_north_u']                   = None
BCs['bc_north_v']                   = None
BCs['bc_north_p']                   = None
BCs['bc_north_T']                   = 300
# numpy.linspace(400, 900, settings['Nodes_x'])

# Initial conditions from previous run/already in memory
Use_inital_values                   = False

# Time advancement
settings['CFL']                     = 0.5
settings['total_time_steps']        = 50
settings['Time_Scheme']             = 'RK4'

print('######################################################')
print('#              2D Navier-Stokes Solver               #')
print('#              Created by J. Mark Epps               #')
print('#          Part of Masters Thesis at UW 2018-2020    #')
print('######################################################\n')
print 'Initializing geometry package...'
#domain=OneDimLine(L,Nx)
domain=TwoDimPlanar(settings)
domain.mesh()
print '################################'

##########################################################################
# -------------------------------------Initialize solver and domain
##########################################################################

print 'Initializing solver package...'
solver=Solvers.TwoDimPlanarSolve(domain, settings, BCs)
print '################################'
print 'Initializing domain...'
if not Use_inital_values:
    T=numpy.zeros((domain.Ny,domain.Nx))
    u=numpy.zeros((domain.Ny,domain.Nx))
    v=numpy.zeros((domain.Ny,domain.Nx))
    p=numpy.zeros((domain.Ny,domain.Nx))
    
    u[:,:]=0
    v[:,:]=0
    
    #domain.rho[:,:]=CP.PropsSI('D','T',300,'P',101325,settings['Fluid'])
    T[:,:]=300
    p[:,:]=101325
    #p=domain.rho*domain.R*T
    domain.rho=p/(domain.R*T)
else:
    domain.rho=rho.copy()
domain.rhou=domain.rho*u
domain.rhov=domain.rho*v
domain.rhoE=domain.rho*(0.5*(u**2+v**2)+domain.Cv*T)

solver.Apply_BCs(domain.rho, domain.rhou, domain.rhov, domain.rhoE, \
                 u, v, p, T, solver.dx, solver.dy)

print '################################'
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
input_file.input_writer(settings, BCs, domain.rho, domain.rhou, domain.rhov, domain.rhoE)
input_file.close()
print '################################'

##########################################################################
# -------------------------------------Solve
##########################################################################
print 'Solving:'
for nt in range(settings['total_time_steps']):
    print 'Time step %i of %i'%(nt+1, settings['total_time_steps'])
    err=solver.Advance_Soln()
    if err==1:
        print '#################### Solver aborted #######################'
        break

#output_file.close()

##########################################################################
# ------------------------------------Post-processing
##########################################################################
u,v,p,T=domain.primitiveFromConserv(domain.rho, domain.rhou, domain.rhov, domain.rhoE)
rho=domain.rho
X,Y=domain.X,domain.Y
# 2D plot
#fig=pyplot.figure(figsize=(7, 7))
#ax = fig.gca(projection='3d')
#ax.plot_surface(domain.X, domain.Y, T, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
##ax.set_xlim(0,0.001)
##ax.set_ylim(0.005,0.006)
#ax.set_zlim(300, BCs['bc_south_T'])
#ax.set_xlabel('$x$ (m)')
#ax.set_ylabel('$y$ (m)')
#ax.set_zlabel('T (K)');
#fig.savefig('Plot1.png',dpi=300)

# 1D Plot
#fig2=pyplot.figure(figsize=(7,7))
#pyplot.plot(domain.Y[:,1]*1000, domain.T[:,1],marker='x')
#pyplot.xlabel('$y$ (mm)')
#pyplot.ylabel('T (K)')
#pyplot.title('Temperature distribution at 2nd x')
#pyplot.xlim(5,6);
#fig2.savefig('Plot2.png',dpi=300)

# Velocity Quiver plot and pressure contour
pl=5
fig3=pyplot.figure(figsize=(7, 7))
pyplot.quiver(X[::pl, ::pl], Y[::pl, ::pl], \
              u[::pl, ::pl], v[::pl, ::pl]) 
pyplot.contourf(X, Y, p-101325, alpha=0.5, cmap=cm.viridis)  
pyplot.colorbar()
pyplot.xlabel('$x$ (m)')
pyplot.ylabel('$y$ (m)')
pyplot.title('Velocity plot and Gage Pressure contours');
#fig3.savefig(datTime+'_Vel_Press.png',dpi=300)

## Temperature contour
#fig4=pyplot.figure(figsize=(7, 7))
#pyplot.contourf(X, Y, T, alpha=0.5, cmap=cm.viridis)  
#pyplot.colorbar()
#pyplot.xlabel('$x$ (m)')
#pyplot.ylabel('$y$ (m)')
#pyplot.title('Temperature distribution');
##fig4.savefig(datTime+'_Temp.png',dpi=300)
#
## Density contour
#fig5=pyplot.figure(figsize=(7, 7))
#pyplot.contourf(X, Y, rho, alpha=0.5, cmap=cm.viridis)  
#pyplot.colorbar()
#pyplot.xlabel('$x$ (m)')
#pyplot.ylabel('$y$ (m)')
#pyplot.title('Density distribution');
##fig4.savefig(datTime+'_Temp.png',dpi=300)

print('Solver has finished its run')