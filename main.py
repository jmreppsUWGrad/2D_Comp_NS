# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:51:15 2018

@author: Joseph
"""

##########################################################################
# ----------------------------------Libraries and classes
##########################################################################
#import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

#from GeomClasses import OneDimLine as OneDimLine
from GeomClasses import TwoDimPlanar as TwoDimPlanar
#import MatClasses as Mat
import SolverClasses as Solvers

##########################################################################
# ------------------------------ Geometry and Domain Setup
##########################################################################
L=3*10**(-3) # Length (x coordinate max)
W=6*10**(-3) # Width (y coordinate max)
Nx=301 # Number of nodes in x
Ny=601 # Number of nodes in y
dt=10**(-6) # Time step size
Nt=1000 # Number of time steps
conv=0.001 # Convergence criteria
mat_prop={'k': 5, 'Cp': 800, 'rho': 8000}

#domain=OneDimLine(L,Nx)
domain2=TwoDimPlanar(L,W,Nx,Ny)

##########################################################################
# --------------------------------------Meshing
##########################################################################
smallest=0.000001
#domain.bias_elem['TwoWayEnd']=smallest
#domain.mesh()
#domain2.xbias_elem['TwoWayEnd']=smallest
domain2.ybias_elem['OneWayUp']=smallest
domain2.mesh()

##########################################################################
# -------------------------------------Initialize domain
##########################################################################
#domain.T[:]=300
domain2.T[:,:]=300
domain2.mat_prop=mat_prop

##########################################################################
# -------------------------------------Boundary conditions
##########################################################################
BC_info={'BCx1': ['F',0,(1,-2)],\
         'BCx2': ['C',(10,300),(1,-2)],\
         'BCy1': ['F',0,(1,-2)],\
         'BCy2': ['F',4*10**8,(1,-299),'C',(10,300),(2,-2)]\
         }
#domain.T[-1]=600 # Boundary condition
#domain2.T[:,0]=600
#domain2.T[:,-1]=300
#domain2.T[0,:]=600
#domain2.T[-1,:]=300

##########################################################################
# -------------------------------------Initialize solver
##########################################################################
#solver=Solvers.OneDimCondSolve(domain,dt,Nt,conv)
solver2=Solvers.TwoDimCondSolve(domain2,dt,Nt,conv)
solver2.BCs=BC_info
solver2.CheckFo() # Check Fo for stability

##########################################################################
# -------------------------------------Solve
##########################################################################
#solver.SolveExpTrans()
#solver.SolveSS()
solver2.SolveExpTrans()
#solver2.SolveSS()

##########################################################################
# ------------------------------------Plots
##########################################################################
#fig=pyplot.figure(figsize=(7, 7), dpi=100)
#ax = fig.gca(projection='3d')
#ax.plot_surface(domain2.X, domain2.Y, domain2.T, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
#ax.set_xlim(0,0.001)
#ax.set_ylim(0.005,0.006)
#ax.set_zlim(300, 700)
#ax.set_xlabel('$x$ (m)')
#ax.set_ylabel('$y$ (m)')
#ax.set_zlabel('T (K)')

fig2=pyplot.figure(figsize=(7,7))
pyplot.plot(domain2.Y[:,1]*1000, domain2.T[:,1],marker='x')
pyplot.xlabel('$y$ (mm)')
pyplot.ylabel('T (K)')
pyplot.title('Temperature distribution at 2nd x')
pyplot.xlim(5,6)

print('End of script')