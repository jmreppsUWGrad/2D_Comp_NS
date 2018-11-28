# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:17:11 2018

@author: Joseph

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

import numpy
#import GeomClasses
#import MatClasses
#import CoolProp.CoolProp as CP
import temporal_schemes

# 1D Solvers (CURRENTLY ONLY FOR CONDUCTION)
class OneDimSolve():
    def __init__(self, geom, timeSize, timeSteps, conv):
        self.Domain=geom # Geometry object
        self.dt=timeSize
        self.Nt=timeSteps
        self.conv=conv
        self.T=self.Domain.T
        self.dx=self.Domain.dx
        self.maxCount=1000
        self.Fo=1.0*self.Domain.mat_prop['k']*self.dt\
        /(self.Domain.mat_prop['rho']*self.Domain.mat_prop['Cp'])
        self.BCs={'BCx1': ('T',600,(0,-1)),\
                 'BCx2': ('T',300,(0,-1)),\
                 'BCy1': ('T',600,(0,-1)),\
                 'BCy2': ('T',300,(0,-1))\
                 }
    
    # Convergence checker
    def CheckConv(self, Tprev, Tnew):
        diff=numpy.sum(numpy.abs(Tnew[:]-Tprev[:]))/numpy.sum(numpy.abs(Tprev[:]))
        print(diff)
        if diff<=self.conv:
            return True
        else:
            return False
    # Solve
    def SolveExpTrans(self):
        Tc=numpy.empty_like(self.T)
        for i in range(self.Nt):
            Tc=self.T.copy()
            self.T[1:-1]=2*self.Fo/(self.dx[:-1]+self.dx[1:])*(Tc[:-2]/self.dx[:-1]+Tc[2:]/self.dx[1:])\
            +(1-2*self.Fo/(self.dx[:-1]+self.dx[1:])*(1/self.dx[:-1]+1/self.dx[1:]))*Tc[1:-1]
        
    def SolveSS(self):
        Tc=numpy.empty_like(self.T)
        count=0
        print 'Residuals:'
        while count<self.maxCount:
            Tc=self.T.copy()
            self.T[1:-1]=(self.dx[1:]*Tc[:-2]+self.dx[:-1]*Tc[2:])\
            /(self.dx[1:]+self.dx[:-1])
            if self.CheckConv(Tc,self.T):
                break

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
        self.dx,self.dy=numpy.meshgrid(geom_obj.dx,geom_obj.dy)
        self.BCs=BCs
    
    # Time step check with dx, dy, T and CFL number
    def getdt(self, T):
#        dx=numpy.sqrt(self.dx**2+self.dy**2)
        
        dx=numpy.zeros_like(self.dx)
        dx[1:-1,1:-1]=0.5*numpy.sqrt((self.dx[1:-1,1:-1]+self.dx[1:-1,:-2])**2+\
                  (self.dy[1:-1,1:-1]+self.dy[:-2,1:-1])**2)
        dx[0,0]      =0.5*numpy.sqrt((self.dx[0,0])**2+(self.dy[0,0])**2)
        dx[0,1:-1]   =0.5*numpy.sqrt((self.dx[0,1:-1]+self.dx[0,:-2])**2+\
                  (self.dy[0,1:-1])**2)
        dx[1:-1,0]   =0.5*numpy.sqrt((self.dx[1:-1,0])**2+\
                  (self.dy[1:-1,0]+self.dy[:-2,0])**2)
        dx[0,-1]     =0.5*numpy.sqrt((self.dx[0,-1])**2+(self.dy[0,-1])**2)
        dx[-1,0]     =0.5*numpy.sqrt((self.dx[-1,0])**2+(self.dy[-1,0])**2)
        dx[-1,1:-1]  =0.5*numpy.sqrt((self.dx[-1,1:-1]+self.dx[-1,:-2])**2+\
                  (self.dy[-1,1:-1])**2)
        dx[1:-1,-1]  =0.5*numpy.sqrt((self.dx[1:-1,-1])**2+(self.dy[1:-1,-1]+\
                  self.dy[:-2,-1])**2)
        dx[-1,-1]    =0.5*numpy.sqrt((self.dx[-1,-1])**2+(self.dy[-1,-1])**2)
#        print(dx)
        c=numpy.sqrt(self.Domain.gamma*self.Domain.R*T) # ADD SPEED OF SOUND RETRIEVAL
#        print(c)
        return numpy.amin(self.CFL*dx/(c))

    # Convergence checker (REMOVE? NO IMPLICIT CALCULATIONS DONE)
    def CheckConv(self, Tprev, Tnew):
        diff=numpy.sum(numpy.abs(Tnew[:]-Tprev[:]))/numpy.sum(numpy.abs(Tprev[:]))
        print(diff)
        if diff<=self.conv:
            return True
        else:
            return False
    # Flux of conservative variables
    # Calculates for entire domain and accounts for periodicity
    # Can be hacked to solve gradients setting rho to 1.0 and u or v to zeros
    def compute_Flux(self, rho, u, v, dx, dy):
        ddx=numpy.empty_like(u)
        ddy=numpy.empty_like(v)
        rhou=rho*u
        rhov=rho*v

        ddx[:,1:-1]=(rhou[:,2:]-rhou[:,:-2])/(dx[:,1:-1]+dx[:,:-2])
        ddy[1:-1,:]=(rhov[2:,:]-rhov[:-2,:])/(dy[1:-1,:]+dy[:-2,:])
        
        if (self.BCs['bc_type_left']=='periodic') or (self.BCs['bc_type_right']=='periodic'):
            ddx[:,0] =(rhou[:,1]-rhou[:,-1])/(dx[:,0]+dx[:,-1])
            ddx[:,-1]=(rhou[:,0]-rhou[:,-2])/(dx[:,-1]+dx[:,0])
        else:
            # Forward/backward differences for boundaries
            ddx[:,0] =(rhou[:,1]-rhou[:,0])/(dx[:,0])
            ddx[:,-1]=(rhou[:,-1]-rhou[:,-2])/(dx[:,-1])
        if (self.BCs['bc_type_north']=='periodic') or (self.BCs['bc_type_south']=='periodic'):
            ddy[0,:] =(rhov[1,:]-rhov[-1,:])/(dy[0,:]+dy[-1,:])
            ddy[-1,:]=(rhov[0,:]-rhov[-2,:])/(dy[-1,:]+dy[0,:])
        else:
            # Forward/backward differences for boundaries
            ddy[0,:] =(rhov[1,:]-rhov[0,:])/(dy[0,:])
            ddy[-1,:]=(rhov[-1,:]-rhov[-2,:])/(dy[-1,:])
        
        return ddx+ddy
    
    # Shear stress gradient calculation for momentum
    def Calculate_Stress(self, u, v, dx, dy):
        mu=self.Domain.mu
        # Central differences up to boundaries
        self.Domain.tau11[1:-1,1:-1]=2.0/3*mu*(2*(u[1:-1,2:]-u[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2])-\
            (v[2:,1:-1]-v[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1]))
        self.Domain.tau12[1:-1,1:-1]=mu*((v[1:-1,2:]-v[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2])+\
            (u[2:,1:-1]-u[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1]))
        self.Domain.tau22[1:-1,1:-1]=2.0/3*mu*(2*(v[2:,1:-1]-v[:-2,1:-1])/(dy[1:-1,1:-1]+dy[:-2,1:-1])-\
            (u[1:-1,2:]-u[1:-1,:-2])/(dx[1:-1,1:-1]+dx[1:-1,:-2]))
        
        # Boundary treatments dependent on periodicity
        if (self.BCs['bc_type_north']=='periodic') or (self.BCs['bc_type_south']=='periodic'):
            # North and south boundary values
            self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
                             (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
            self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
                             (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
            
            self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
                             (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
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
            
            self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,0])/dx[1:-1,0]+\
                             (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
            self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,-1]-v[1:-1,-2])/dx[1:-1,-1]+\
                             (u[2:,-1]-u[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1]))
            
            self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0])-\
                             (u[1:-1,1]-u[1:-1,0])/dx[1:-1,0])
            self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1])-\
                             (u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1])
            # Corner treatments
            self.Domain.tau11[0,0]    =2.0/3*mu*(2*(u[0,1]-u[0,0])/(dx[0,0])-\
                             (v[1,0]-v[-1,0])/(dy[0,0]+dy[-1,0]))
            self.Domain.tau11[0,-1]   =2.0/3*mu*(2*(u[0,-1]-u[0,-2])/(dx[0,-1])-\
                             (v[1,-1]-v[-1,-1])/(dy[0,-1]+dy[-1,-1]))
            self.Domain.tau11[-1,0]   =2.0/3*mu*(2*(u[-1,1]-u[-1,0])/(dx[-1,0])-\
                             (v[0,0]-v[-2,0])/(dy[-1,0]+dy[-2,0]))
            self.Domain.tau11[-1,-1]  =2.0/3*mu*(2*(u[-1,-1]-u[-1,-2])/(dx[-1,-1])-\
                             (v[0,-1]-v[-2,-1])/(dy[-1,-1]+dy[-2,-1]))
            
            self.Domain.tau12[0,0]    =mu*((v[0,1]-v[0,0])/dx[0,0]+\
                             (u[1,0]-u[-1,0])/(dy[0,0]+dy[-1,0]))
            self.Domain.tau12[0,-1]   =mu*((v[0,-1]-v[0,-2])/(dx[0,-1])+\
                             (u[1,-1]-u[-1,-1])/(dy[0,-1]+dy[-1,-1]))
            self.Domain.tau12[-1,0]   =mu*((v[-1,1]-v[-1,0])/dx[-1,0]+\
                             (u[0,0]-u[-2,0])/(dy[-1,0]+dy[-2,0]))
            self.Domain.tau12[-1,-1]  =mu*((v[-1,-1]-v[-1,-2])/dx[-1,-1]+\
                             (u[0,-1]-u[-2,-1])/(dy[-1,-1]+dy[-2,-1]))
            
            self.Domain.tau22[0,0]    =2.0/3*mu*(2*(v[1,0]-v[-1,0])/(dy[0,0]+dy[-1,0])-\
                             (u[0,1]-u[0,0])/dx[0,0])
            self.Domain.tau22[0,-1]   =2.0/3*mu*(2*(v[1,-1]-v[-1,-1])/(dy[0,-1]+dy[-1,-1])-\
                             (u[0,-1]-u[0,-2])/dx[0,-1])
            self.Domain.tau22[-1,0]   =2.0/3*mu*(2*(v[0,0]-v[-2,0])/(dy[-1,0]+dy[-2,0])-\
                             (u[-1,1]-u[-1,0])/dx[-1,0])
            self.Domain.tau22[-1,-1]  =2.0/3*mu*(2*(v[0,-1]-v[-2,-1])/(dy[-1,-1]+dy[-2,-1])-\
                             (u[-1,-1]-u[-1,-2])/dx[-1,-1])
            
            
        elif (self.BCs['bc_type_left']=='periodic') or (self.BCs['bc_type_right']=='periodic'):
            # Left/right boundaries
            self.Domain.tau11[1:-1,0] =2.0/3*mu*(2*(u[1:-1,1]-u[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1])-\
                             (v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
            self.Domain.tau11[1:-1,-1]=2.0/3*mu*(2*(u[1:-1,0]-u[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2])-\
                             (v[2:,-1]-v[:-2,-1])/(dy[1:-1,0]+dy[:-2,0]))
            
            self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1])+\
                             (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
            self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,0]-v[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2])+\
                             (u[2:,-1]-u[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1]))
            
            self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[:-2,0])/(dy[1:-1,0]+dy[:-2,0])-\
                             (u[1:-1,1]-u[1:-1,-1])/(dx[1:-1,0]+dx[1:-1,-1]))
            self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[:-2,-1])/(dy[1:-1,-1]+dy[:-2,-1])-\
                             (u[1:-1,0]-u[1:-1,-2])/(dx[1:-1,-1]+dx[1:-1,-2]))
            # North and south boundary values
            self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
                             (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
            self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
                             (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
            
            self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
                             (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
            self.Domain.tau12[-1,1:-1]=mu*((v[-1,2:]-v[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])+\
                             (u[-1,1:-1]-u[-2,1:-1])/dy[-1,1:-1])
            
            self.Domain.tau22[0,1:-1] =2.0/3*mu*(2*(v[1,1:-1]-v[0,1:-1])/dy[0,1:-1]-\
                             (u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2]))
            self.Domain.tau22[-1,1:-1]=2.0/3*mu*(2*(v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1]-\
                             (u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2]))
            # Corner treatments
            self.Domain.tau11[0,0]    =2.0/3*mu*(2*(u[0,1]-u[0,-1])/(dx[0,0]+dx[0,-1])-\
                             (v[1,0]-v[0,0])/dy[0,0])
            self.Domain.tau11[0,-1]   =2.0/3*mu*(2*(u[0,0]-u[0,-2])/(dx[0,-1]+dx[0,-2])-\
                             (v[1,-1]-v[0,-1])/dy[0,-1])
            self.Domain.tau11[-1,0]   =2.0/3*mu*(2*(u[-1,1]-u[-1,-1])/(dx[-1,0]+dx[-1,-1])-\
                             (v[-1,0]-v[-2,0])/dy[-1,0])
            self.Domain.tau11[-1,-1]  =2.0/3*mu*(2*(u[-1,0]-u[-1,-2])/(dx[-1,-1]+dx[-1,-2])-\
                             (v[-1,-1]-v[-2,-1])/dy[-1,-1])
            
            self.Domain.tau12[0,0]    =mu*((v[0,1]-v[0,-1])/(dx[0,0]+dx[0,-1])+\
                             (u[1,0]-u[0,0])/dy[0,0])
            self.Domain.tau12[0,-1]   =mu*((v[0,0]-v[0,-2])/(dx[-1,-1]+dx[-1,-2])+\
                             (u[1,-1]-u[0,-1])/dy[0,-1])
            self.Domain.tau12[-1,0]   =mu*((v[-1,1]-v[-1,-1])/(dx[-1,0]+dx[-1,-1])+\
                             (u[-1,0]-u[-2,0])/dy[-1,0])
            self.Domain.tau12[-1,-1]  =mu*((v[-1,0]-v[-1,-2])/(dx[-1,-1]+dx[-1,-2])+\
                             (u[-1,-1]-u[-2,-1])/dy[-1,-1])
            
            self.Domain.tau22[0,0]    =2.0/3*mu*(2*(v[1,0]-v[0,0])/dy[0,0]-\
                             (u[0,1]-u[0,-1])/(dx[0,0]+dx[0,-1]))
            self.Domain.tau22[0,-1]   =2.0/3*mu*(2*(v[1,-1]-v[0,-1])/dy[0,-1]-\
                             (u[0,0]-u[0,-2])/(dx[0,-1]+dx[0,-2]))
            self.Domain.tau22[-1,0]   =2.0/3*mu*(2*(v[-1,0]-v[-2,0])/dy[-1,0]-\
                             (u[-1,1]-u[-1,-1])/(dx[-1,0]+dx[-1,-1]))
            self.Domain.tau22[-1,-1]  =2.0/3*mu*(2*(v[-1,-1]-v[-2,-1])/dy[-1,-1]-\
                             (u[-1,0]-u[-1,-2])/(dx[-1,-1]+dx[-1,-2]))
        
        # No periodicity
        else:
            # North and south boundary values
            self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,:-2])/(dx[0,1:-1]+dx[0,:-2])-\
                             (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
            self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,:-2])/(dx[-1,1:-1]+dx[-1,:-2])-\
                             (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
            
            if self.BCs['bc_type_south']=='outlet':
                self.Domain.tau12[0,1:-1] =self.Domain.tau12[1,1:-1]
            elif self.BCs['bc_type_south']=='slip_wall':
                self.Domain.tau12[0,1:-1] =0
            else:
                self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,:-2])/(dx[0,1:-1]+dx[0,:-2])+\
                             (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
            if self.BCs['bc_type_north']=='outlet':
                self.Domain.tau12[-1,1:-1]=self.Domain.tau12[-2,1:-1]
            elif self.BCs['bc_type_north']=='slip_wall':
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
            
            if self.BCs['bc_type_left']=='outlet':
                self.Domain.tau12[1:-1,0] =self.Domain.tau12[1:-1,1]
            elif self.BCs['bc_type_left']=='slip_wall':
                self.Domain.tau12[1:-1,0] =0
            else:
                self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,0])/dx[1:-1,0]+\
                                 (u[2:,0]-u[:-2,0])/(dy[1:-1,0]+dy[:-2,0]))
            if self.BCs['bc_type_right']=='outlet':
                self.Domain.tau12[1:-1,-1]=self.Domain.tau12[1:-1,-2]
            elif self.BCs['bc_type_right']=='slip_wall':
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
            
    
    # Work via control surface calculation (grad*(sigma*v))
    def Source_CSWork(self, u, v, dx, dy):
        tau11u=self.Domain.tau11*u
        tau12v=self.Domain.tau12*v
        tau21u=self.Domain.tau12*u
        tau22v=self.Domain.tau22*v
        
        work =self.compute_Flux(1.0, tau11u, tau21u, dx, dy)
        work+=self.compute_Flux(1.0, tau12v, tau22v, dx, dy)
        
        return work
    
    # Heat conduction gradient source term
    def Source_Cond(self, T, dx, dy):
        qx=numpy.empty_like(T)
        qy=numpy.empty_like(T)
        k=self.Domain.k
        # Central difference
        qx[:,1:-1]=-k*(T[:,2:]-T[:,:-2])/(dx[:,1:-1]+dx[:,:-2])
        qy[1:-1,:]=-k*(T[2:,:]-T[:-2,:])/(dy[1:-1,:]+dy[:-2,:])
        # Forward/backward difference for boundaries (if not periodic)
        if (self.BCs['bc_type_left']=='periodic') or (self.BCs['bc_type_right']=='periodic'):        
            qx[:,0] =-k*(T[:,1]-T[:,-1])/(dx[:,0]+dx[:,-1])
            qx[:,-1]=-k*(T[:,0]-T[:,-2])/(dx[:,-1]+dx[:,-2])
        elif self.BCs['bc_type_left']=='outlet':
            qx[:,0] =qx[:,1]
            qx[:,-1]=-k*(T[:,-1]-T[:,-2])/dx[:,-1]
        elif self.BCs['bc_type_right']=='outlet':
            qx[:,0] =-k*(T[:,1]-T[:,0])/dx[:,0]
            qx[:,-1]=qx[:,-2]
        else:
            qx[:,0] =-k*(T[:,1]-T[:,0])/dx[:,0]
            qx[:,-1]=-k*(T[:,-1]-T[:,-2])/dx[:,-1]
        
        if (self.BCs['bc_type_north']=='periodic') or (self.BCs['bc_type_south']=='periodic'):
            qy[0,:] =-k*(T[1,:]-T[-1,:])/(dy[0,:]+dy[-1,:])
            qy[-1,:]=-k*(T[0,:]-T[-2,:])/(dy[-1,:]+dy[-2,:])
        elif self.BCs['bc_type_north']=='outlet':
            qy[-1,:]=qy[-2,:]
            qy[0,:] =-k*(T[1,:]-T[0,:])/dy[0,:]
        elif self.BCs['bc_type_south']=='outlet':
            qy[0,:] =qy[1,:]
            qy[-1,:]=-k*(T[-1,:]-T[-2,:])/dy[-1,:]
        else:
            qy[0,:] =-k*(T[1,:]-T[0,:])/dy[0,:]
            qy[-1,:]=-k*(T[-1,:]-T[-2,:])/dy[-1,:]
        return self.compute_Flux(1.0,qx,qy,dx,dy)
    
    # Bondary condition handler (not including periodic BCs)
    def Apply_BCs(self, rho, rhou, rhov, rhoE, u, v, p, T, dx, dy):
        # Start with wall BCs
        
        # Left face
        if self.BCs['bc_type_left']=='wall':
#            print 'Left: wall'
            p[:,0]  =p[:,1]
            rhou[:,0]  =0
            rhov[:,0]  =0
            if (type(self.BCs['bc_left_T']) is str)\
                and (self.BCs['bc_left_T']=='zero_grad'):
                T[:,0]  =T[:,1]
            elif type(self.BCs['bc_left_T']) is tuple:
                T[:,0]  =T[:,1]-self.BCs['bc_left_T'][1]*dx[:,0]
            else:
                T[:,0]  =self.BCs['bc_left_T']
            rho[:,0]=p[:,0]/(self.Domain.R*T[:,0])
            rhoE[:,0]=rho[:,0]*self.Domain.Cv*T[:,0]
        
        elif self.BCs['bc_type_left']=='slip_wall':
            rhou[:,0]  =0
            p[:,0]  =p[:,1]
            if (type(self.BCs['bc_left_T']) is str)\
                and (self.BCs['bc_left_T']=='zero_grad'):
                T[:,0]  =T[:,1]
            elif type(self.BCs['bc_left_T']) is tuple:
                T[:,0]  =T[:,1]-self.BCs['bc_left_T'][1]*dx[:,0]
            else:
                T[:,0]  =self.BCs['bc_left_T']
            rho[:,0]=p[:,0]/(self.Domain.R*T[:,0])
            rhoE[:,0]=rho[:,0]*self.Domain.Cv*T[:,0]
            
        elif self.BCs['bc_type_left']=='inlet':
            p[:,0]  =self.BCs['bc_left_p']
#            pt      =self.BCs['bc_left_p']
            u[:,0]  =self.BCs['bc_left_u']
            v[:,0]  =self.BCs['bc_left_v']
            if (type(self.BCs['bc_left_T']) is str)\
                and (self.BCs['bc_left_T']=='zero_grad'):
                T[:,0]  =T[:,1]
            else:
                T[:,0]  =self.BCs['bc_left_T']
            
#            u[:,0]=numpy.sqrt(2*self.Domain.gamma*self.Domain.R*T[:,0]/(self.Domain.gamma-1)\
#                 *((p[:,0]/pt)**(self.Domain.gamma/(self.Domain.gamma-1))-1))
#            p[:,0]=pt*(1+(self.Domain.gamma-1)/2*u[:,0]/(self.Domain.gamma*self.Domain.R*T[:,0]))\
#                 **((self.Domain.gamma-1)/self.Domain.gamma)

#            p[:,0]=rho[:,0]*self.Domain.R*T[:,0]
            
            rhou[:,0]=rho[:,0]*u[:,0]
            rhov[:,0]=rho[:,0]*v[:,0]
#            rhoE[:,0]=p[:,0]/(self.Domain.gamma-1)+rho[:,0]*0.5*(u[:,0]**2+v[:,0]**2)
            rhoE[:,0]=rho[:,0]*(0.5*(u[:,0]**2+v[:,0]**2)+self.Domain.Cv*T[:,0])
                
        elif self.BCs['bc_type_left']=='outlet':
            p[:,0]=self.BCs['bc_left_p']
            rhoE[:,0]=p[:,0]/(self.Domain.gamma-1)+rho[:,0]*0.5*(u[:,0]**2+v[:,0]**2)
        
        # Periodic boundary for Poiseuille flow        
        elif self.BCs['bc_left_p']!=None:
            p[:,0]=self.BCs['bc_left_p']
            rhoE[:,0]=p[:,0]/(self.Domain.gamma-1)+rho[:,0]*0.5*(u[:,0]**2+v[:,0]**2)
        
        # Right face
        if self.BCs['bc_type_right']=='wall':
#            print 'Right: wall'
            p[:,-1]  =p[:,-2]
            rhou[:,-1]  =0
            rhov[:,-1]  =0
            if (type(self.BCs['bc_right_T']) is str)\
                and (self.BCs['bc_right_T']=='zero_grad'):
                T[:,-1]  =T[:,-2]
            elif type(self.BCs['bc_right_T']) is tuple:
                T[:,-1]  =T[:,-2]+self.BCs['bc_right_T'][1]*dx[:,-1]
            else:
                T[:,-1]  =self.BCs['bc_right_T']
            rho[:,-1]=p[:,-1]/(self.Domain.R*T[:,-1])
            rhoE[:,-1]=rho[:,-1]*self.Domain.Cv*T[:,-1]
            
        elif self.BCs['bc_type_right']=='slip_wall':
            p[:,-1]  =p[:,-2]
            rhou[:,-1]  =0
            if (type(self.BCs['bc_right_T']) is str)\
                and (self.BCs['bc_right_T']=='zero_grad'):
                T[:,-1]  =T[:,-2]
            elif type(self.BCs['bc_right_T']) is tuple:
                T[:,-1]  =T[:,-2]+self.BCs['bc_right_T'][1]*dx[:,-1]
            else:
                T[:,-1]  =self.BCs['bc_right_T']
            rho[:,-1]=p[:,-1]/(self.Domain.R*T[:,-1])
            rhoE[:,-1]=rho[:,-1]*self.Domain.Cv*T[:,-1]
        
        elif self.BCs['bc_type_right']=='inlet':
            u[:,-1]  =self.BCs['bc_right_u']
            v[:,-1]  =self.BCs['bc_right_v']
            if (type(self.BCs['bc_right_T']) is str)\
                and (self.BCs['bc_right_T']=='zero_grad'):
                T[:,-1]  =T[:,-2]
            else:
                T[:,-1]  =self.BCs['bc_right_T']
            p[:,-1]  =self.BCs['bc_right_p']
#            rho[:,-1]=p[:,-1]/self.Domain.R/T[:,-1]
            
            rhou[:,-1]=rho[:,-1]*u[:,-1]
            rhov[:,-1]=rho[:,-1]*v[:,-1]
#            rhoE[:,-1]=p[:,-1]/(self.Domain.gamma-1)+rho[:,-1]*0.5*(u[:,-1]**2+v[:,-1]**2)
            rhoE[:,-1]=rho[:,-1]*(0.5*(u[:,-1]**2+v[:,-1]**2)+self.Domain.Cv*T[:,-1])
                
        elif self.BCs['bc_type_right']=='outlet':
            p[:,-1]=self.BCs['bc_right_p']
            rhoE[:,-1]=p[:,-1]/(self.Domain.gamma-1)+rho[:,-1]*0.5*(u[:,-1]**2+v[:,-1]**2)
        
        elif self.BCs['bc_right_p']!=None:
            p[:,-1]=self.BCs['bc_right_p']
            rhoE[:,-1]=p[:,-1]/(self.Domain.gamma-1)+rho[:,-1]*0.5*(u[:,-1]**2+v[:,-1]**2)
            
        # South face
        if self.BCs['bc_type_south']=='wall':
#            print 'South: wall'
            p[0,:]  =p[1,:]
            rhou[0,:]  =0
            rhov[0,:]  =0
            if (type(self.BCs['bc_south_T']) is str)\
                and (self.BCs['bc_south_T']=='zero_grad'):
                T[0,:]  =T[1,:]
            elif type(self.BCs['bc_south_T']) is tuple:
                T[0,:]  =T[1,:]-self.BCs['bc_south_T'][1]*dy[0,:]
            else:
                T[0,:]  =self.BCs['bc_south_T']
            rho[0,:]=p[0,:]/(self.Domain.R*T[0,:])
            rhoE[0,:]=rho[0,:]*self.Domain.Cv*T[0,:]
            
        elif self.BCs['bc_type_south']=='slip_wall':
            p[0,:]  =p[1,:]
            rhov[0,:]  =0
            if (type(self.BCs['bc_south_T']) is str)\
                and (self.BCs['bc_south_T']=='zero_grad'):
                T[0,:]  =T[1,:]
            elif type(self.BCs['bc_south_T']) is tuple:
                T[0,:]  =T[1,:]-self.BCs['bc_south_T'][1]*dy[0,:]
            else:
                T[0,:]  =self.BCs['bc_south_T']
            rho[0,:]=p[0,:]/(self.Domain.R*T[0,:])
            rhoE[0,:]=rho[0,:]*self.Domain.Cv*T[0,:]
        
        elif self.BCs['bc_type_south']=='inlet':
            u[0,:]  =self.BCs['bc_south_u']
            v[0,:]  =self.BCs['bc_south_v']
            if (type(self.BCs['bc_south_T']) is str)\
                and (self.BCs['bc_south_T']=='zero_grad'):
                T[0,:]  =T[1,:]
            else:
                T[0,:]  =self.BCs['bc_south_T']
            p[0,:]  =self.BCs['bc_south_p']
#            rho[0,:]=p[0,:]/self.Domain.R/T[0,:]
            
            rhou[0,:]=rho[0,:]*u[0,:]
            rhov[0,:]=rho[0,:]*v[0,:]
#            rhoE[0,:]=p[0,:]/(self.Domain.gamma-1)+rho[0,:]*0.5*(u[0,:]**2+v[0,:]**2)
            rhoE[0,:]=rho[0,:]*(0.5*(u[0,:]**2+v[0,:]**2)+self.Domain.Cv*T[0,:])
                
        elif self.BCs['bc_type_south']=='outlet':
            p[0,:]=self.BCs['bc_south_p']
            rhoE[0,:]=p[0,:]/(self.Domain.gamma-1)+rho[0,:]*0.5*(u[0,:]**2+v[0,:]**2)
        
        # Periodic boundary for Poiseuille flow        
        elif self.BCs['bc_south_p']!=None:
            p[0,:]=self.BCs['bc_south_p']
            rhoE[0,:]=p[0,:]/(self.Domain.gamma-1)+rho[0,:]*0.5*(u[0,:]**2+v[0,:]**2)
            
        # North face
        if self.BCs['bc_type_north']=='wall':
#            print 'North: wall'
            p[-1,:]  =p[-2,:]
            rhou[-1,:]  =0
            rhov[-1,:]  =0
            if (type(self.BCs['bc_north_T']) is str)\
                and (self.BCs['bc_north_T']=='zero_grad'):
                T[-1,:]  =T[-2,:]
            elif type(self.BCs['bc_north_T']) is tuple:
                T[-1,:]  =T[-2,:]+self.BCs['bc_north_T'][1]*dy[-1,:]
            else:
                T[-1,:]  =self.BCs['bc_north_T']
            rho[-1,:]=p[-1,:]/(self.Domain.R*T[-1,:])
            rhoE[-1,:]=rho[-1,:]*self.Domain.Cv*T[-1,:]
            
        elif self.BCs['bc_type_north']=='slip_wall':
            p[-1,:]  =p[-2,:]
            rhov[-1,:]  =0
            if (type(self.BCs['bc_north_T']) is str)\
                and (self.BCs['bc_north_T']=='zero_grad'):
                T[-1,:]  =T[-2,:]
            elif type(self.BCs['bc_north_T']) is tuple:
                T[-1,:]  =T[-2,:]+self.BCs['bc_north_T'][1]*dy[-1,:]
            else:
                T[-1,:]  =self.BCs['bc_north_T']
            rho[-1,:]=p[-1,:]/(self.Domain.R*T[-1,:])
        
        elif self.BCs['bc_type_north']=='inlet':
            u[-1,:]  =self.BCs['bc_north_u']
            v[-1,:]  =self.BCs['bc_north_v']
            if (type(self.BCs['bc_north_T']) is str)\
                and (self.BCs['bc_north_T']=='zero_grad'):
                T[-1,:]  =T[-2,:]
            else:
                T[-1,:]  =self.BCs['bc_north_T']
            p[-1,:]  =self.BCs['bc_north_p']
#            rho[-1,:]=p[-1,:]/self.Domain.R/T[-1,:]
            
            rhou[-1,:]=rho[-1,:]*u[-1,:]
            rhov[-1,:]=rho[-1,:]*v[-1,:]
#            rhoE[-1,:]=p[-1,:]/(self.Domain.gamma-1)+0.5*rho[-1,:]*(u[-1,:]**2+v[-1,:]**2)
            rhoE[-1,:]=rho[-1,:]*(0.5*(u[-1,:]**2+v[-1,:]**2)+self.Domain.Cv*T[-1,:])
                
        elif self.BCs['bc_type_north']=='outlet':
            p[-1,:]=self.BCs['bc_north_p']
            rhoE[-1,:]=p[-1,:]/(self.Domain.gamma-1)+0.5*rho[-1,:]*(u[-1,:]**2+v[-1,:]**2)
        
        # Periodic boundary for Poiseuille flow        
        elif self.BCs['bc_north_p']!=None:
            p[-1,:]=self.BCs['bc_north_p']
            rhoE[-1,:]=p[-1,:]/(self.Domain.gamma-1)+0.5*rho[-1,:]*(u[-1,:]**2+v[-1,:]**2)
        
    # Main compressible solver (1 time step)
    def Advance_Soln(self):
        rho_0=self.Domain.rho.copy()
        rhou_0=self.Domain.rhou.copy()
        rhov_0=self.Domain.rhov.copy()
        rhoE_0=self.Domain.rhoE.copy()
        rho_c=rho_0.copy()
        rhou_c=rhou_0.copy()
        rhov_c=rhov_0.copy()
        rhoE_c=rhoE_0.copy()
                
        if self.time_scheme=='Euler':
            rk_coeff = numpy.array([1,0])
            rk_substep_fraction = numpy.array([1,0])
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
        dt=self.getdt(T)
        if (numpy.isnan(dt)) or (dt<=0):
            print 'Time step size: %f'%dt
            print '*********Diverging time step***********'
            return 1, dt
        print 'Time step size: %.6f'%dt
        
        for step in range(Nstep):
            ###################################################################
            # Compute primitive variables
            ###################################################################
            u,v,p,T=self.Domain.primitiveFromConserv(rho_c, rhou_c, rhov_c, rhoE_c)
            
            ###################################################################
            # Compute time deriviatives of conservatives (2nd order central schemes)
            ###################################################################
            # Calculate shear stress arrays for momentum and energy
            self.Calculate_Stress(u, v, self.dx, self.dy)
    
            # Density
            drhodt[step] =-self.compute_Flux(rho_c, u, v, self.dx, self.dy)
    
            # x-momentum (flux, pressure, shear stress, gravity)
            drhoudt[step] =-self.compute_Flux(rhou_c, u, v, self.dx, self.dy)
            drhoudt[step]-=self.compute_Flux(1.0, p, numpy.zeros_like(v), self.dx, self.dy)
            drhoudt[step]+=self.compute_Flux(1.0, self.Domain.tau11, self.Domain.tau12, self.dx, self.dy)
            drhoudt[step]+=rho_c*self.gx
    
            # y-momentum (flux, pressure, shear stress, gravity)
            drhovdt[step] =-self.compute_Flux(rhov_c, u, v, self.dx, self.dy)
            drhovdt[step]-=self.compute_Flux(1.0, numpy.zeros_like(u), p, self.dx, self.dy)
            drhovdt[step]+=self.compute_Flux(1.0, self.Domain.tau12, self.Domain.tau22, self.dx, self.dy)
            drhovdt[step]+=rho_c*self.gy
            
            # Energy (flux, pressure-work, shear-work, conduction, gravity)
            drhoEdt[step] =-self.compute_Flux(rhoE_c, u, v, self.dx, self.dy)
            drhoEdt[step]-=self.compute_Flux(p, u, v, self.dx, self.dy)
            drhoEdt[step]+=self.Source_CSWork(u, v, self.dx, self.dy)
            drhoEdt[step]-=self.Source_Cond(T, self.dx, self.dy)
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
                self.Apply_BCs(rho_c, rhou_c, rhov_c, rhoE_c, u, v, p, T, self.dx, self.dy)
                
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
        self.Apply_BCs(self.Domain.rho, self.Domain.rhou, self.Domain.rhov,\
                       self.Domain.rhoE, u, v, p, T, self.dx, self.dy)
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
        
        if (numpy.isnan(numpy.amax(self.Domain.rho))) or \
            (numpy.isnan(numpy.amax(self.Domain.rhou))) or \
            (numpy.isnan(numpy.amax(self.Domain.rhov))) or \
            (numpy.isnan(numpy.amax(self.Domain.rhoE))):
            print '**************Divergence detected****************'
            return 1, dt
        
        ###################################################################
        # Output data to file?????
        ###################################################################
        
        
        
        return 0, dt