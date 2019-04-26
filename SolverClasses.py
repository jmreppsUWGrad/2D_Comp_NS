# -*- coding: utf-8 -*-
"""
######################################################
#       2D Compressible Navier-Stokes Solver         #
#              Created by J. Mark Epps               #
#          Part of Masters Thesis at UW 2018-2020    #
######################################################

Solver classes for Compressible N-S equations. Takes in given object (geometry),
time step and convergence information and alters the object's temperature, 
velocity, pressure, density. BCs are applied as appropriate, but must be 
defined and copied into the solver object.

Assumptions:
    -equal discretization spacings in either x or y
    -constant thermal conductivity for conduction
    -constant viscosity for shear stress

Features:
    -Conservative Fourrier number correction based on smallest discretization
    -

p/pt=(1+(gamma-1)/2*Ma**2)**(-gamma/(gamma-1))

"""

import numpy as np
import BCClasses
#import MatClasses
#import CoolProp.CoolProp as CP
import temporal_schemes

# 2D solver
class TwoDimPlanarSolve():
    def __init__(self, geom_obj, settings, BCs):
        self.Domain=geom_obj # Geometry object
        self.CFL=settings['CFL']
        self.time_scheme=settings['Time_Scheme']
#        self.Nt=settings['total_time_steps']
#        self.conv=conv
        self.gx=settings['Gravity_x']
        self.gy=settings['Gravity_y']
        self.dx,self.dy=np.meshgrid(geom_obj.dx,geom_obj.dy)
        
        # BC class
        self.BCs=BCClasses.BCs(settings, BCs, self.dx, self.dy)
    
    # Get eigenvalues of current time step
    def eigenval(self, u, v, T):
        c=np.sqrt(self.Domain.gamma*self.Domain.R*T) # ADD SPEED OF SOUND RETRIEVAL
        lam1=max(np.amax(u-c),np.amax(v-c))
        lam2=u
        lam3=v
        lam4=max(np.amax(u+c),np.amax(v+c))
        return max(lam1,0), np.amax(lam2), np.amax(lam3), lam4
    
    # Time step check with dx, dy, T and CFL number
    def getdt(self, lam1, lam2, lam3, lam4, T):
        dt1=np.amin(self.CFL*self.dx/(max(lam1,lam2,lam3,lam4)))
        dt2=np.amin(self.CFL*self.dy/(max(lam1,lam2,lam3,lam4)))
        
        return min(dt1,dt2)
    
    # Interpolation function (become flux function?)
    def interpolate(self, k1, k2, func):
        if func=='Linear':
            return 0.5*k1+0.5*k2
        else:
            return 2*k1*k2/(k1+k2)
    
    # coefficients for velocity weighting in stress calculation
    def get_Coeff(self, dx, dy, k, mult_fac):
        aW=np.zeros_like(dx)
        aE=np.zeros_like(dx)
        aS=np.zeros_like(dx)
        aN=np.zeros_like(dx)
        
        # Left/right face factors
        aW[1:-1,1:-1] =0.5*k\
                    *(dy[1:-1,1:-1]+dy[:-2,1:-1])/(dx[1:-1,:-2])
        aE[1:-1,1:-1] =0.5*k\
                    *(dy[1:-1,1:-1]+dy[:-2,1:-1])/(dx[1:-1,1:-1])
            # At north/south bondaries
        aW[0,1:-1]    =0.5*k\
            *(dy[0,1:-1])/(dx[0,:-2])
        aE[0,1:-1]    =0.5*k\
            *(dy[0,1:-1])/(dx[0,1:-1])
        aW[-1,1:-1]   =0.5*k\
            *(dy[-1,1:-1])/(dx[-1,:-2])
        aE[-1,1:-1]   =0.5*k\
            *(dy[-1,1:-1])/(dx[-1,1:-1])
            # At east/west boundaries
        aE[0,0]       =0.5*k\
            *(dy[0,0])/dx[0,0]
        aE[1:-1,0]    =0.5*k\
            *(dy[1:-1,0]+dy[:-2,0])/dx[1:-1,0]
        aE[-1,0]      =0.5*k\
            *(dy[-1,0])/dx[-1,0]
        aW[0,-1]      =0.5*k\
            *(dy[0,-1])/dx[0,-1]
        aW[1:-1,-1]   =0.5*k\
            *(dy[1:-1,-1]+dy[:-2,-1])/dx[1:-1,-1]
        aW[-1,-1]     =0.5*k\
            *(dy[-1,-1])/dx[-1,-1]
        
        # Top/bottom faces
        aS[1:-1,1:-1]=0.5*k\
            *(dx[1:-1,1:-1]+dx[1:-1,:-2])/dy[:-2,1:-1]
        aN[1:-1,1:-1]=0.5*k\
            *(dx[1:-1,1:-1]+dx[1:-1,:-2])/dy[1:-1,1:-1]
        
            # Area account for east/west boundary nodes
        aS[1:-1,0]    =0.5*k\
            *(dx[1:-1,0])/(dy[:-2,0])
        aN[1:-1,0]    =0.5*k\
            *(dx[1:-1,0])/(dy[1:-1,0])
        aS[1:-1,-1]   =0.5*k\
            *(dx[1:-1,-1])/(dy[:-2,-1])
        aN[1:-1,-1]   =0.5*k\
            *(dx[1:-1,-1])/(dy[1:-1,-1])
            # Forward/backward difference for north/south boundaries
        aN[0,0]       =0.5*k\
            *dx[0,0]/dy[0,0]
        aN[0,1:-1]    =0.5*k\
            *(dx[0,1:-1]+dx[0,:-2])/dy[0,1:-1]
        aN[0,-1]      =0.5*k\
            *dx[0,-1]/dy[0,-1]
        aS[-1,0]      =0.5*k\
            *dx[-1,0]/dy[-1,0]
        aS[-1,1:-1]   =0.5*k\
            *(dx[0,1:-1]+dx[0,:-2])/dy[-1,1:-1]
        aS[-1,-1]     =0.5*k\
            *dx[-1,-1]/dy[-1,-1]
        
        return aW,aE,aS,aN
    
    # Flux of conservative variables
    # Calculates for entire domain and accounts for periodicity
    # Can be hacked to solve gradients setting rho to variable and u or v to zeros
    def compute_Flux(self, rho, u, v, dx, dy, hx, hy, lam, scheme):
        ddx=np.zeros_like(u)
        ddy=np.zeros_like(v)
#        rhou=rho*u
#        rhov=rho*v
        if scheme=='UDS':
            ud=1.0
            LLF=0
        else:
            ud=0
            LLF=1.0
        
        # Flux across left/right faces
        ddx[:,1:]  =-0.5*(rho[:,1:]*(u[:,1:]-ud*np.abs(u[:,1:]))+\
               rho[:,:-1]*(u[:,:-1]+ud*np.abs(u[:,:-1]))-LLF*lam*(u[:,1:]-u[:,:-1]))/hx[:,1:]
        ddx[:,:-1]+=0.5*(rho[:,:-1]*(u[:,:-1]+ud*np.abs(u[:,:-1]))+\
               rho[:,1:]*(u[:,1:]-ud*np.abs(u[:,1:]))-LLF*lam*(u[:,1:]-u[:,:-1]))/hx[:,:-1]
        
            # North/south boundaries (OLD)
#        ddx[0,1:]   =-Ax[0,1:]*0.5*(rho[0,1:]*(u[0,1:]-np.abs(u[0,1:]))+rho[0,:-1]*(u[0,:-1]+np.abs(u[0,:-1]))\
#               -lam*(u[0,1:]-u[0,:-1]))
#        ddx[0,:-1] +=Ax[0,:-1]*0.5*(rho[0,:-1]*(u[0,:-1]+np.abs(u[0,:-1]))+rho[0,1:]*(u[0,1:]-np.abs(u[0,1:]))\
#               -lam*(u[0,1:]-u[0,:-1]))
#        ddx[-1,1:]  =-Ax[-1,1:]*0.5*(rho[-1,1:]*(u[-1,1:]-np.abs(u[-1,1:]))+rho[1,:-1]*(u[-1,:-1]+np.abs(u[-1,:-1]))\
#               -lam*(u[-1,1:]-u[-1,:-1]))
#        ddx[-1,:-1]+=Ax[-1,:-1]*0.5*(rho[-1,:-1]*(u[-1,:-1]+np.abs(u[-1,:-1]))+rho[1,1:]*(u[-1,1:]-np.abs(u[-1,1:]))\
#               -lam*(u[-1,1:]-u[-1,:-1]))
            # Flux across left/right faces, no area consideration
        
        # Flux across top/bottom faces
        ddy[1:,:]  =0.5*(rho[1:,:]*(v[1:,:]-ud*np.abs(v[1:,:]))+\
               rho[:-1,:]*(v[:-1,:]+ud*np.abs(v[:-1,:]))-LLF*lam*(v[1:,:]-v[:-1,:]))/hy[1:,:]
        ddy[:-1,:]+=0.5*(rho[:-1,:]*(v[:-1,:]+ud*np.abs(v[:-1,:]))+\
               rho[1:,:]*(v[1:,:]-ud*np.abs(v[1:,:]))-LLF*lam*(v[1:,:]-v[:-1,:]))/hy[:-1,:]
            # East/west boundaries (OLD)
#        ddy[1:,0]   =-Ay[1:,0]*0.5*(rho[1:,0]*(v[1:,0]-np.abs(v[1:,0]))+rho[:-1,0]*(v[:-1,0]+np.abs(v[:-1,0]))\
#               -lam*(v[1:,0]-v[:-1,0]))
#        ddy[:-1,0] +=Ay[:-1,0]*0.5*(rho[:-1,0]*(v[:-1,0]+np.abs(v[:-1,0]))+rho[1:,0]*(v[1:,0]-np.abs(v[1:,0]))\
#               -lam*(v[1:,0]-v[:-1,0]))
#        ddy[1:,-1]  =-Ay[1:,-1]*0.5*(rho[1:,-1]*(v[1:,-1]-np.abs(v[1:,-1]))+rho[:-1,-1]*(v[:-1,-1]+np.abs(v[:-1,-1]))\
#               -lam*(v[1:,-1]-v[:-1,-1]))
#        ddy[:-1,-1]+=Ay[:-1,-1]*0.5*(rho[:-1,-1]*(v[:-1,-1]+np.abs(v[:-1,-1]))+rho[1:,-1]*(v[1:,-1]-np.abs(v[1:,-1]))\
#               -lam*(v[1:,-1]-v[:-1,-1]))
        
        
#        if (self.BCs.BCs['bc_type_left']=='periodic') or (self.BCs.BCs['bc_type_right']=='periodic'):
#            ddx[:,0] =(rhou[:,1]-rhou[:,-1])/(dx[:,0]+dx[:,-1])
#            ddx[:,-1]=(rhou[:,0]-rhou[:,-2])/(dx[:,-1]+dx[:,0])
#        else:
#            # Forward/backward differences for boundaries
#            ddx[:,0] =(rhou[:,1]-rhou[:,0])/(dx[:,0])
#            ddx[:,-1]=(rhou[:,-1]-rhou[:,-2])/(dx[:,-1])
#        if (self.BCs.BCs['bc_type_north']=='periodic') or (self.BCs.BCs['bc_type_south']=='periodic'):
#            ddy[0,:] =(rhov[1,:]-rhov[-1,:])/(dy[0,:]+dy[-1,:])
#            ddy[-1,:]=(rhov[0,:]-rhov[-2,:])/(dy[-1,:]+dy[0,:])
#        else:
#            # Forward/backward differences for boundaries
#            ddy[0,:] =(rhov[1,:]-rhov[0,:])/(dy[0,:])
#            ddy[-1,:]=(rhov[-1,:]-rhov[-2,:])/(dy[-1,:])
        
        return ddx+ddy
    
    # Shear stress calculation
    def Calculate_Stress(self, u, v, dx, dy):
        mu=self.Domain.mu
        	# Central differences up to boundaries
        self.Domain.tau11[1:-1,1:-1]=2.0/3*mu*(2*(u[1:-1,2:]-u[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2])-\
           (v[2:,1:-1]-v[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1]))
        self.Domain.tau12[1:-1,1:-1]=mu*((v[1:-1,2:]-v[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2])+\
           (u[2:,1:-1]-u[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1]))
        self.Domain.tau22[1:-1,1:-1]=2.0/3*mu*(2*(v[2:,1:-1]-v[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1])-\
           (u[1:-1,2:]-u[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2]))
       
#        # Boundary treatments dependent on periodicity
#        if (self.BCs.BCs['bc_type_north']=='periodic') or (self.BCs.BCs['bc_type_south']=='periodic'):
#           # North and south boundary values
#           self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
#                            (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
#           self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
#                            (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
#           
#           self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
#                            (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
#           self.Domain.tau12[-1,1:-1]=mu*((v[-1,2:]-v[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])+\
#                            (u[-1,1:-1]-u[-2,1:-1])/dy[-1,1:-1])
#           
#           self.Domain.tau22[0,1:-1] =2.0/3*mu*(2*(v[1,1:-1]-v[0,1:-1])/dy[0,1:-1]-\
#                            (u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2]))
#           self.Domain.tau22[-1,1:-1]=2.0/3*mu*(2*(v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1]-\
#                            (u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2]))
#           # Left/right boundaries
#           self.Domain.tau11[1:-1,0] =2.0/3*mu*(2*(u[1:-1,1]-u[1:-1,0])/dx[1:-1,0]-\
#                            (v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
#           self.Domain.tau11[1:-1,-1]=2.0/3*mu*(2*(u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1]-\
#                            (v[2:,-1]-v[:-2,-1])/(dy[1:-1,0]+dy[:-2,0]))
#           
#           self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,0])/dx[1:-1,0]+\
#                            (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
#           self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,-1]-v[1:-1,-2])/dx[1:-1,-1]+\
#                            (u[2:,-1]-u[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1]))
#           
#           self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0])-\
#                            (u[1:-1,1]-u[1:-1,0])/dx[1:-1,0])
#           self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1])-\
#                            (u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1])
#           # Corner treatments
#           self.Domain.tau11[0,0]    =2.0/3*mu*(2*(u[0,1]-u[0,0])/(dx[0,0])-\
#                            (v[1,0]-v[-1,0])/(dy[0,0]+dy[-1,0]))
#           self.Domain.tau11[0,-1]   =2.0/3*mu*(2*(u[0,-1]-u[0,-2])/(dx[0,-1])-\
#                            (v[1,-1]-v[-1,-1])/(dy[0,-1]+dy[-1,-1]))
#           self.Domain.tau11[-1,0]   =2.0/3*mu*(2*(u[-1,1]-u[-1,0])/(dx[-1,0])-\
#                            (v[0,0]-v[-2,0])/(dy[-1,0]+dy[-2,0]))
#           self.Domain.tau11[-1,-1]  =2.0/3*mu*(2*(u[-1,-1]-u[-1,-2])/(dx[-1,-1])-\
#                            (v[0,-1]-v[-2,-1])/(dy[-1,-1]+dy[-2,-1]))
#           
#           self.Domain.tau12[0,0]    =mu*((v[0,1]-v[0,0])/dx[0,0]+\
#                            (u[1,0]-u[-1,0])/(dy[0,0]+dy[-1,0]))
#           self.Domain.tau12[0,-1]   =mu*((v[0,-1]-v[0,-2])/(dx[0,-1])+\
#                            (u[1,-1]-u[-1,-1])/(dy[0,-1]+dy[-1,-1]))
#           self.Domain.tau12[-1,0]   =mu*((v[-1,1]-v[-1,0])/dx[-1,0]+\
#                            (u[0,0]-u[-2,0])/(dy[-1,0]+dy[-2,0]))
#           self.Domain.tau12[-1,-1]  =mu*((v[-1,-1]-v[-1,-2])/dx[-1,-1]+\
#                            (u[0,-1]-u[-2,-1])/(dy[-1,-1]+dy[-2,-1]))
#           
#           self.Domain.tau22[0,0]    =2.0/3*mu*(2*(v[1,0]-v[-1,0])/(dy[0,0]+dy[-1,0])-\
#                            (u[0,1]-u[0,0])/dx[0,0])
#           self.Domain.tau22[0,-1]   =2.0/3*mu*(2*(v[1,-1]-v[-1,-1])/(dy[0,-1]+dy[-1,-1])-\
#                            (u[0,-1]-u[0,-2])/dx[0,-1])
#           self.Domain.tau22[-1,0]   =2.0/3*mu*(2*(v[0,0]-v[-2,0])/(dy[-1,0]+dy[-2,0])-\
#                            (u[-1,1]-u[-1,0])/dx[-1,0])
#           self.Domain.tau22[-1,-1]  =2.0/3*mu*(2*(v[0,-1]-v[-2,-1])/(dy[-1,-1]+dy[-2,-1])-\
#                            (u[-1,-1]-u[-1,-2])/dx[-1,-1])
#           
#           
#        elif (self.BCs.BCs['bc_type_left']=='periodic') or (self.BCs.BCs['bc_type_right']=='periodic'):
#           # Left/right boundaries
#           self.Domain.tau11[1:-1,0] =2.0/3*mu*(2*(u[1:-1,1]-u[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1])-\
#                            (v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
#           self.Domain.tau11[1:-1,-1]=2.0/3*mu*(2*(u[1:-1,0]-u[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2])-\
#                            (v[2:,-1]-v[:-2,-1])/(dy[1:-1,0]+dy[:-2,0]))
#           
#           self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1])+\
#                            (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
#           self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,0]-v[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2])+\
#                            (u[2:,-1]-u[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1]))
#           
#           self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0])-\
#                            (u[1:-1,1]-u[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1]))
#           self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1])-\
#                            (u[1:-1,0]-u[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2]))
#           # North and south boundary values
#           self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
#                            (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
#           self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
#                            (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
#           
#           self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
#                            (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
#           self.Domain.tau12[-1,1:-1]=mu*((v[-1,2:]-v[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])+\
#                            (u[-1,1:-1]-u[-2,1:-1])/dy[-1,1:-1])
#           
#           self.Domain.tau22[0,1:-1] =2.0/3*mu*(2*(v[1,1:-1]-v[0,1:-1])/dy[0,1:-1]-\
#                            (u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2]))
#           self.Domain.tau22[-1,1:-1]=2.0/3*mu*(2*(v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1]-\
#                            (u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2]))
#           # Corner treatments
#           self.Domain.tau11[0,0]    =2.0/3*mu*(2*(u[0,1]-u[0,-1])/(dx[0,0]+dx[0,-1])-\
#                            (v[1,0]-v[0,0])/dy[0,0])
#           self.Domain.tau11[0,-1]   =2.0/3*mu*(2*(u[0,0]-u[0,-2])/(dx[0,-1]+dx[0,-2])-\
#                            (v[1,-1]-v[0,-1])/dy[0,-1])
#           self.Domain.tau11[-1,0]   =2.0/3*mu*(2*(u[-1,1]-u[-1,-1])/(dx[-1,0]+dx[-1,-1])-\
#                            (v[-1,0]-v[-2,0])/dy[-1,0])
#           self.Domain.tau11[-1,-1]  =2.0/3*mu*(2*(u[-1,0]-u[-1,-2])/(dx[-1,-1]+dx[-1,-2])-\
#                            (v[-1,-1]-v[-2,-1])/dy[-1,-1])
#           
#           self.Domain.tau12[0,0]    =mu*((v[0,1]-v[0,-1])/(dx[0,0]+dx[0,-1])+\
#                            (u[1,0]-u[0,0])/dy[0,0])
#           self.Domain.tau12[0,-1]   =mu*((v[0,0]-v[0,-2])/(dx[-1,-1]+dx[-1,-2])+\
#                            (u[1,-1]-u[0,-1])/dy[0,-1])
#           self.Domain.tau12[-1,0]   =mu*((v[-1,1]-v[-1,-1])/(dx[-1,0]+dx[-1,-1])+\
#                            (u[-1,0]-u[-2,0])/dy[-1,0])
#           self.Domain.tau12[-1,-1]  =mu*((v[-1,0]-v[-1,-2])/(dx[-1,-1]+dx[-1,-2])+\
#                            (u[-1,-1]-u[-2,-1])/dy[-1,-1])
#           
#           self.Domain.tau22[0,0]    =2.0/3*mu*(2*(v[1,0]-v[0,0])/dy[0,0]-\
#                            (u[0,1]-u[0,-1])/(dx[0,0]+dx[0,-1]))
#           self.Domain.tau22[0,-1]   =2.0/3*mu*(2*(v[1,-1]-v[0,-1])/dy[0,-1]-\
#                            (u[0,0]-u[0,-2])/(dx[0,-1]+dx[0,-2]))
#           self.Domain.tau22[-1,0]   =2.0/3*mu*(2*(v[-1,0]-v[-2,0])/dy[-1,0]-\
#                            (u[-1,1]-u[-1,-1])/(dx[-1,0]+dx[-1,-1]))
#           self.Domain.tau22[-1,-1]  =2.0/3*mu*(2*(v[-1,-1]-v[-2,-1])/dy[-1,-1]-\
#                            (u[-1,0]-u[-1,-2])/(dx[-1,-1]+dx[-1,-2]))
       
        # No periodicity
#        else:
           # North and south boundary values
        self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
                        (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
        self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
                        (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
           
        if self.BCs.BCs['bc_type_south']=='outlet':
           self.Domain.tau12[0,1:-1] =self.Domain.tau12[1,1:-1]
        elif self.BCs.BCs['bc_type_south']=='slip_wall':
           self.Domain.tau12[0,1:-1] =0
        else:
           self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
                        (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
        if self.BCs.BCs['bc_type_north']=='outlet':
           self.Domain.tau12[-1,1:-1]=self.Domain.tau12[-2,1:-1]
        elif self.BCs.BCs['bc_type_north']=='slip_wall':
           self.Domain.tau12[-1,1:-1]=0
        else:
           self.Domain.tau12[-1,1:-1]=mu*((v[-1,2:]-v[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])+\
                            (u[-1,1:-1]-u[-2,1:-1])/dy[-1,1:-1])
           
        self.Domain.tau22[0,1:-1] =2.0/3*mu*(2*(v[1,1:-1]-v[0,1:-1])/dy[0,1:-1]-\
                        (u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2]))
        self.Domain.tau22[-1,1:-1]=2.0/3*mu*(2*(v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1]-\
                        (u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2]))
        # Left/right boundaries
        self.Domain.tau11[1:-1,0] =2.0/3*mu*(2*(u[1:-1,1]-u[1:-1,0])/dx[1:-1,0]-\
                        (v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
        self.Domain.tau11[1:-1,-1]=2.0/3*mu*(2*(u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1]-\
                        (v[2:,-1]-v[:-2,-1])/(dy[1:-1,0]+dy[:-2,0]))
           
        if self.BCs.BCs['bc_type_left']=='outlet':
           self.Domain.tau12[1:-1,0] =self.Domain.tau12[1:-1,1]
        elif self.BCs.BCs['bc_type_left']=='slip_wall':
           self.Domain.tau12[1:-1,0] =0
        else:
           self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,0])/dx[1:-1,0]+\
                            (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
        if self.BCs.BCs['bc_type_right']=='outlet':
           self.Domain.tau12[1:-1,-1]=self.Domain.tau12[1:-1,-2]
        elif self.BCs.BCs['bc_type_right']=='slip_wall':
           self.Domain.tau12[1:-1,-1]=0
        else:
           self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,-1]-v[1:-1,-2])/dx[1:-1,-1]+\
                            (u[2:,-1]-u[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1]))
           
        self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0])-\
                        (u[1:-1,1]-u[1:-1,0])/dx[1:-1,0])
        self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1])-\
                        (u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1])
        # Corner treatments
        self.Domain.tau11[0,0]    =2.0/3*mu*(2*(u[0,1]-u[0,0])/(dx[0,0])-\
                        (v[1,0]-v[0,0])/dy[0,0])
        self.Domain.tau11[0,-1]   =2.0/3*mu*(2*(u[0,-1]-u[0,-2])/(dx[0,-1])-\
                        (v[1,-1]-v[0,-1])/dy[0,-1])
        self.Domain.tau11[-1,0]   =2.0/3*mu*(2*(u[-1,1]-u[-1,0])/(dx[-1,0])-\
                        (v[-1,0]-v[-2,0])/dy[-1,0])
        self.Domain.tau11[-1,-1]  =2.0/3*mu*(2*(u[-1,-1]-u[-1,-2])/(dx[-1,-1])-\
                        (v[-1,-1]-v[-2,-1])/dy[-1,-1])
           
        self.Domain.tau12[0,0]    =mu*((v[0,1]-v[0,0])/dx[0,0]+\
                        (u[1,0]-u[0,0])/dy[0,0])
        self.Domain.tau12[0,-1]   =mu*((v[0,-1]-v[0,-2])/(dx[0,-1])+\
                        (u[1,-1]-u[0,-1])/dy[0,-1])
        self.Domain.tau12[-1,0]   =mu*((v[-1,1]-v[-1,0])/dx[-1,0]+\
                        (u[-1,0]-u[-2,0])/dy[-1,0])
        self.Domain.tau12[-1,-1]  =mu*((v[-1,-1]-v[-1,-2])/dx[-1,-1]+\
                        (u[-1,-1]-u[-2,-1])/dy[-1,-1])
           
        self.Domain.tau22[0,0]    =2.0/3*mu*(2*(v[1,0]-v[0,0])/dy[0,0]-\
                        (u[0,1]-u[0,0])/dx[0,0])
        self.Domain.tau22[0,-1]   =2.0/3*mu*(2*(v[1,-1]-v[0,-1])/dy[0,-1]-\
                        (u[0,-1]-u[0,-2])/dx[0,-1])
        self.Domain.tau22[-1,0]   =2.0/3*mu*(2*(v[-1,0]-v[-2,0])/dy[-1,0]-\
                        (u[-1,1]-u[-1,0])/dx[-1,0])
        self.Domain.tau22[-1,-1]  =2.0/3*mu*(2*(v[-1,-1]-v[-2,-1])/dy[-1,-1]-\
                        (u[-1,-1]-u[-1,-2])/dx[-1,-1])
           
        self.BCs.Visc_BCs(self.Domain.tau11, self.Domain.tau12, self.Domain.tau22,\
                         np.ones_like(self.dx), np.ones_like(self.dx))
       
    # Work via control surface calculation (grad*(sigma*v))
    def Source_CSWork(self, u, v, dx, dy, hx, hy):
        tau11u=self.Domain.tau11*u
    	tau12u=self.Domain.tau12*u
    	tau21v=self.Domain.tau12*v
    	tau22v=self.Domain.tau22*v
    		
    	work =self.compute_Flux(np.ones_like(dx), tau11u, tau12u, dx, dy, hx, hy, 0, 'LLF')
    	work+=self.compute_Flux(np.ones_like(dx), tau21v, tau22v, dx, dy, hx, hy, 0, 'LLF')
        return work
    
    # Heat conduction gradient source term
    def Source_Cond(self, T, dx, dy, hx, hy):
        qx=np.zeros_like(T)
        qy=np.zeros_like(T)
#        aW=np.zeros_like(dx)
#        aE=np.zeros_like(dx)
#        aS=np.zeros_like(dx)
#        aN=np.zeros_like(dx)
        k=self.Domain.k
        # Temperature weighting coefficients
        
        # Left/right face factors
#        aW[1:-1,1:-1] =0.5*k\
#                    *(dy[1:-1,1:-1]+dy[:-2,1:-1])/(dx[1:-1,:-2])
#        aE[1:-1,1:-1] =0.5*k\
#                    *(dy[1:-1,1:-1]+dy[:-2,1:-1])/(dx[1:-1,1:-1])
#            # At north/south bondaries
#        aW[0,1:-1]    =0.5*k\
#            *(dy[0,1:-1])/(dx[0,:-2])
#        aE[0,1:-1]    =0.5*k\
#            *(dy[0,1:-1])/(dx[0,1:-1])
#        aW[-1,1:-1]   =0.5*k\
#            *(dy[-1,1:-1])/(dx[-1,:-2])
#        aE[-1,1:-1]   =0.5*k\
#            *(dy[-1,1:-1])/(dx[-1,1:-1])
#            # At east/west boundaries
#        aE[0,0]       =0.5*k\
#            *(dy[0,0])/dx[0,0]
#        aE[1:-1,0]    =0.5*k\
#            *(dy[1:-1,0]+dy[:-2,0])/dx[1:-1,0]
#        aE[-1,0]      =0.5*k\
#            *(dy[-1,0])/dx[-1,0]
#        aW[0,-1]      =0.5*k\
#            *(dy[0,-1])/dx[0,-1]
#        aW[1:-1,-1]   =0.5*k\
#            *(dy[1:-1,-1]+dy[:-2,-1])/dx[1:-1,-1]
#        aW[-1,-1]     =0.5*k\
#            *(dy[-1,-1])/dx[-1,-1]
#        
#        # Top/bottom faces
#        aS[1:-1,1:-1]=0.5*k\
#            *(dx[1:-1,1:-1]+dx[1:-1,:-2])/dy[:-2,1:-1]
#        aN[1:-1,1:-1]=0.5*k\
#            *(dx[1:-1,1:-1]+dx[1:-1,:-2])/dy[1:-1,1:-1]
#        
#            # Area account for east/west boundary nodes
#        aS[1:-1,0]    =0.5*k\
#            *(dx[1:-1,0])/(dy[:-2,0])
#        aN[1:-1,0]    =0.5*k\
#            *(dx[1:-1,0])/(dy[1:-1,0])
#        aS[1:-1,-1]   =0.5*k\
#            *(dx[1:-1,-1])/(dy[:-2,-1])
#        aN[1:-1,-1]   =0.5*k\
#            *(dx[1:-1,-1])/(dy[1:-1,-1])
#            # Forward/backward difference for north/south boundaries
#        aN[0,0]       =0.5*k\
#            *dx[0,0]/dy[0,0]
#        aN[0,1:-1]    =0.5*k\
#            *(dx[0,1:-1]+dx[0,:-2])/dy[0,1:-1]
#        aN[0,-1]      =0.5*k\
#            *dx[0,-1]/dy[0,-1]
#        aS[-1,0]      =0.5*k\
#            *dx[-1,0]/dy[-1,0]
#        aS[-1,1:-1]   =0.5*k\
#            *(dx[0,1:-1]+dx[0,:-2])/dy[-1,1:-1]
#        aS[-1,-1]     =0.5*k\
#            *dx[-1,-1]/dy[-1,-1]
#        
#        qx[:,1:]    = aW[:,1:]*T[:,:-1]
#        qx[:,0]     = aE[:,0]*T[:,1]
#        
#        qx[:,1:-1] += aE[:,1:-1]*T[:,2:]
#        qx[1:,:]   += aS[1:,:]*T[:-1,:]
#        qx[:-1,:]  += aN[:-1,:]*T[1:,:]
#        qx         -= (aW+aE+aS+aN)*T
        
        qx[:,1:]   += (k*(T[:,:-1]-T[:,1:])/dx[:,:-1])/hx[:,1:]
        qx[:,:-1]  += (k*(T[:,1:]-T[:,:-1])/dx[:,:-1])/hx[:,:-1]
        
        qy[1:,:]   += (k*(T[:-1,:]-T[1:,:])/dy[:-1,:])/hy[1:,:]
        qy[:-1,:]  += (k*(T[1:,:]-T[:-1,:])/dy[:-1,:])/hy[:-1,:]
        
        # Apply boundary conditions on heat flux
        self.BCs.Visc_BCs(np.ones_like(qx),np.ones_like(qx),np.ones_like(qx),qx,qy)
#        if self.BCs.BCs['bc_type_left']=='outlet':
#            qx[:,0] -=k*(T[:,1]-T[:,0])/dx[:,0] # Effect is 0 flux in x
#        if self.BCs.BCs['bc_type_right']=='outlet':
#            qx[:,-1]-=k*(T[:,-1]-T[:,-2])/dx[:,-1]
#        if self.BCs.BCs['bc_type_north']=='outlet':
#            qx[-1,:]-=k(T[-1,:]-T[-2,:])/dy[-1,:]
#        if self.BCs.BCs['bc_type_south']=='outlet':
#            qx[0,:] -=k*(T[1,:]-T[0,:])/dy[0,:]
#        if (self.BCs.BCs['bc_type_left']=='periodic') or (self.BCs.BCs['bc_type_right']=='periodic'):        
#            qx[:,0] =-k*(T[:,1]-T[:,-1])/(dx[:,0]+dx[:,-1])
#            qx[:,-1]=-k*(T[:,0]-T[:,-2])/(dx[:,-1]+dx[:,-2])
           
#        elif type(self.BCs.BCs['bc_left_T']) is tuple:
#            qx[:,0] =self.BCs.BCs['bc_left_T'][1]
#            qx[:,-1]=-k*(T[:,-1]-T[:,-2])/dx[:,-1]
        
#        elif type(self.BCs.BCs['bc_right_T']) is tuple:
#            qx[:,0] =-k*(T[:,1]-T[:,0])/dx[:,0]
#            qx[:,-1]=self.BCs.BCs['bc_right_T'][1]
#        else:
#            qx[:,0] =-k*(T[:,1]-T[:,0])/dx[:,0]
#            qx[:,-1]=-k*(T[:,-1]-T[:,-2])/dx[:,-1]
        
#        if (self.BCs.BCs['bc_type_north']=='periodic') or (self.BCs.BCs['bc_type_south']=='periodic'):
#            qy[0,:] =-k*(T[1,:]-T[-1,:])/(dy[0,:]+dy[-1,:])
#            qy[-1,:]=-k*(T[0,:]-T[-2,:])/(dy[-1,:]+dy[-2,:])
        
#        elif type(self.BCs.BCs['bc_north_T']) is tuple:
#            qx[-1,:]=self.BCs.BCs['bc_north_T'][1]
#            qx[0,:] =-k*(T[1,:]-T[0,:])/dy[0,:]
        
#        elif type(self.BCs.BCs['bc_south_T']) is tuple:
#            qx[0,:] =self.BCs.BCs['bc_south_T'][1]
#            qx[-1,:]=-k*(T[-1,:]-T[-2,:])/dy[-1,:]
#        else:
#            qx[0,:] =-k*(T[1,:]-T[0,:])/dy[0,:]
#            qx[-1,:]=-k*(T[-1,:]-T[-2,:])/dy[-1,:]
        
        return qx+qy
    
    # Main compressible solver (1 time step)
    def Advance_Soln(self, hx, hy):
        rho_0=self.Domain.rho.copy()
        rhou_0=self.Domain.rhou.copy()
        rhov_0=self.Domain.rhov.copy()
        rhoE_0=self.Domain.rhoE.copy()
        rho_c=rho_0.copy()
        rhou_c=rhou_0.copy()
        rhov_c=rhov_0.copy()
        rhoE_c=rhoE_0.copy()
                
        if self.time_scheme=='Euler':
            rk_coeff = np.array([1,0])
            rk_substep_fraction = np.array([1,0])
            Nstep = 1
            drhodt =[0]*Nstep
            drhoudt=[0]*Nstep
            drhovdt=[0]*Nstep
            drhoEdt=[0]*Nstep
            
        else:
            RK_info=temporal_schemes.runge_kutta(self.time_scheme)
            Nstep = RK_info.Nk
            if Nstep<0:
                return 1, -1 # Scheme not recognized; abort solver
            rk_coeff = RK_info.rk_coeff
            rk_substep_fraction = RK_info.rk_substep_fraction

            drhodt =[0]*Nstep
            drhoudt=[0]*Nstep
            drhovdt=[0]*Nstep
            drhoEdt=[0]*Nstep
        
        u,v,p,T=self.Domain.primitiveFromConserv(rho_0, rhou_0, rhov_0, rhoE_0)
        lam1,lam2,lam3,lam4=self.eigenval(u,v,T)
        dt=self.getdt(lam1,lam2,lam3,lam4,T)
        if (np.isnan(dt)) or (dt<=0):
#            print '    Time step size: %f'%dt
            print '*********Diverging time step***********'
            return 1, dt
#        print '    Time step size: %.6f'%dt
        lam1,lam2,lam3,lam4=0,0,0,0 # If not using eigenvalues for flux
        for step in range(Nstep):
            ###################################################################
            # Compute primitive variables
            ###################################################################
            u,v,p,T=self.Domain.primitiveFromConserv(rho_c, rhou_c, rhov_c, rhoE_c)
            
            ###################################################################
            # Compute time deriviatives of conservatives (2nd order central schemes)
            ###################################################################
            # Calculate stress
            self.Calculate_Stress(u, v, self.dx, self.dy)
			
            # Density
            drhodt[step] =self.compute_Flux(rho_c, u, v, self.dx, self.dy, hx, hy, lam1, 'UDS')
            
            # x-momentum (flux, pressure, shear stress, gravity)
            drhoudt[step] =self.compute_Flux(rhou_c, u, v, self.dx, self.dy, hx, hy, lam2, 'UDS')
            drhoudt[step]+=self.compute_Flux(p, np.ones_like(u), np.zeros_like(v), self.dx, self.dy, hx, hy, 0, 'LLF')
            drhoudt[step]+=self.compute_Flux(np.ones_like(u), self.Domain.tau11, self.Domain.tau12, self.dx, self.dy, hx, hy, 0, 'LLF')
            drhoudt[step]+=rho_c*self.gx
            
            # y-momentum (flux, pressure, shear stress, gravity)
            drhovdt[step] =self.compute_Flux(rhov_c, u, v, self.dx, self.dy, hx, hy, lam3, 'UDS')
            drhovdt[step]+=self.compute_Flux(p, np.zeros_like(u), np.ones_like(v), self.dx, self.dy, hx, hy, 0, 'LLF')
            drhovdt[step]+=self.compute_Flux(np.ones_like(v), self.Domain.tau12, self.Domain.tau22, self.dx, self.dy, hx, hy, 0, 'LLF')
            drhovdt[step]+=rho_c*self.gy
            
            # Energy (flux, pressure-work, shear-work, conduction, gravity)
            drhoEdt[step] =self.compute_Flux(rhoE_c, u, v, self.dx, self.dy, hx, hy, lam4, 'UDS')
            drhoEdt[step]+=self.compute_Flux(p, u, v, self.dx, self.dy, hx, hy, 0, 'LLF')
            drhoEdt[step]+=self.Source_CSWork(u, v, self.dx, self.dy, hx, hy)
            drhoEdt[step]-=self.Source_Cond(T, self.dx, self.dy, hx, hy)
            drhoEdt[step]+=rho_c*(self.gx*u + self.gy*v)

            # Compute intermediate conservative values for RK stepping
            rho_c =rho_0.copy()
            rhou_c=rhou_0.copy()
            rhov_c=rhov_0.copy()
            rhoE_c=rhoE_0.copy()
            
            if step < (Nstep - 1):
                for rk_index in range(step + 1):
                    
                    rho_c += dt*rk_coeff[step+1][rk_index]*drhodt[rk_index]
                    rhou_c+= dt*rk_coeff[step+1][rk_index]*drhoudt[rk_index]
                    rhov_c+= dt*rk_coeff[step+1][rk_index]*drhovdt[rk_index]
                    rhoE_c+= dt*rk_coeff[step+1][rk_index]*drhoEdt[rk_index]
            
                ###################################################################
                # Apply boundary conditions
                ###################################################################
#                self.Apply_BCs(rho_c, rhou_c, rhov_c, rhoE_c, u, v, p, T, self.dx, self.dy)
                self.BCs.Apply_BCs(rho_c, rhou_c, rhov_c, rhoE_c, u, v, p, T)
                
                # Experiment-rectangular solid inside domain, border on south face
#                u[25:35,25:35]=0
#                v[25:35,25:35]=0
#                T[25:35,25:35]=600
#                p[25:35,25]=p[25:35,24]
#                p[25:35,35]=p[25:35,36]
#                p[35,25:35]=p[36,25:35]
#                p[25,25:35]=p[24,25:35]
#                
#                rho_c[25:35,25:35]=p[25:35,25:35]/self.Domain.R/T[25:35,25:35]
#                rhou_c[25:35,25:35]=rho_c[25:35,25:35]*u[25:35,25:35]
#                rhov_c[25:35,25:35]=rho_c[25:35,25:35]*v[25:35,25:35]
#                rhoE_c[25:35,25:35]=rho_c[25:35,25:35]*0.5*(u[25:35,25:35]**2+v[25:35,25:35]**2+2*self.Domain.Cv*T[25:35,25:35])
    
            ###################################################################
            # END OF TIME STEP CALCULATIONS
            ###################################################################
            
        ###################################################################
        # Compute new conservative values at new time step
        ###################################################################
        for step in range(Nstep):    
            self.Domain.rho += dt * rk_substep_fraction[step] * drhodt[step]
            self.Domain.rhou+= dt * rk_substep_fraction[step] * drhoudt[step]
            self.Domain.rhov+= dt * rk_substep_fraction[step] * drhovdt[step]
            self.Domain.rhoE+= dt * rk_substep_fraction[step] * drhoEdt[step]

        ###################################################################
        # Apply boundary conditions
        ###################################################################
#        self.Apply_BCs(self.Domain.rho, self.Domain.rhou, self.Domain.rhov,\
#                       self.Domain.rhoE, u, v, p, T, self.dx, self.dy)
        self.BCs.Apply_BCs(self.Domain.rho, self.Domain.rhou, self.Domain.rhov,\
                       self.Domain.rhoE, u, v, p, T)
        # Experiment-rectangular solid inside domain, border on south face
#        u[:10,20:30]=0
#        v[:10,20:30]=0
#        T[:10,20:30]=600
#        p[:10,20]=p[:10,19]
#        p[:10,30]=p[:10,31]
#        p[10,20:30]=p[11,20:30]
#        p[1:9,21:29]=101325
#        
#        self.Domain.rho[:10,20:30]=p[:10,20:30]/self.Domain.R/T[:10,20:30]
#        self.Domain.rhou[:10,20:30]=self.Domain.rho[:10,20:30]*u[:10,20:30]
#        self.Domain.rhov[:10,20:30]=self.Domain.rho[:10,20:30]*v[:10,20:30]
#        self.Domain.rhoE[:10,20:30]=self.Domain.rho[:10,20:30]*0.5*(u[:10,20:30]**2+v[:10,20:30]**2+2*self.Domain.Cv*T[:10,20:30])
    
        ###################################################################
        # Divergence check
        ###################################################################
        
        if (np.amin(self.Domain.rho)<=0) or (np.isnan(np.amax(self.Domain.rho))):
            print '********* Divergence detected - Density **********'
            return 2, dt
        elif (np.isnan(np.amax(self.Domain.rhou))):
            print '********* Divergence detected - x-momentum ********'
            return 3, dt
        elif (np.isnan(np.amax(self.Domain.rhov))):
            print '********* Divergence detected - y-momentum ********'
            return 4, dt
        elif (np.amax(self.Domain.rhoE)<=0):
            print '********* Divergence detected - Energy **********'
            return 5, dt
        else:
            return 0, dt
        