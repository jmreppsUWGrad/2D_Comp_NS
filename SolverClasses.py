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


"""

import numpy
#import GeomClasses
#import MatClasses
import CoolProp.CoolProp as CP
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
        self.dx,self.dy=numpy.meshgrid(geom_obj.dx,geom_obj.dy)
        self.BCs=BCs
    
    # Time step check with all dx and dys for stability (CHANGE TO CFL NUMBER CHECK)
    def getdt(self):
        dx=min(numpy.amin(self.dx),numpy.amin(self.dy))
        c=numpy.sqrt(self.Domain.gamma*self.Domain.R*numpy.amax(self.Domain.T)) # ADD SPEED OF SOUND RETRIEVAL
        return self.CFL*dx/(c)

    # Convergence checker (REMOVE? NO IMPLICIT CALCULATIONS DONE)
    def CheckConv(self, Tprev, Tnew):
        diff=numpy.sum(numpy.abs(Tnew[:]-Tprev[:]))/numpy.sum(numpy.abs(Tprev[:]))
        print(diff)
        if diff<=self.conv:
            return True
        else:
            return False
    # Solve
    """ To do:
        - flux terms calculator
        - source terms
        - RK time advancement (eventually)
        
    """
    # Flux computer (flux of conservative variables AND gradient calculations)
    # Calculates for entire domain assuming periodicity in both dimensions
    # (BCs other than Periodicity is applied after calculating new values)
    def compute_Flux(self, rho, u, v, dx, dy):
        dx*=2 # Central difference schemes
        dy*=2
        ddx=numpy.empty_like(u)
        ddy=numpy.empty_like(v)
        rhou=rho*u
        rhov=rho*v
#        ddx=(rhou[1:-1,2:]-rhou[1:-1,:-2])/dx[1:-1,1:-1]
#        ddy=(rhov[2:,1:-1]-rhov[:-2,1:-1])/dy[1:-1,1:-1]

        ddx[:,1:-1]=(rhou[:,2:]-rhou[:,:-2])/dx[:,1:-1]
        ddy[1:-1,:]=(rhov[2:,:]-rhov[:-2,:])/dy[1:-1,:]
        
        ddx[:,0] =(rhou[:,1]-rhou[:,-1])/dx[:,0]
        ddx[:,-1]=(rhou[:,0]-rhou[:,-2])/dx[:,-1]
        
        ddy[0,:] =(rhov[1,:]-rhov[-1,:])/dy[0,:]
        ddy[-1,:]=(rhov[0,:]-rhov[-2,:])/dy[-1,:]
        
        dx/=2.0 # Reset original discretizations
        dy/=2.0
        
        return ddx+ddy
    
    # Shear stress gradient calculation for momentum
    def Calculate_Stress(self, u, v, dx, dy):
        dx*=2 # Central difference schemes
        dy*=2
        mu=self.Domain.mu
        # Central differences up to boundaries
        self.Domain.tau11[1:-1,1:-1]=2.0/3*mu*(2*(u[1:-1,2:]-u[1:-1,:-2])/dx[1:-1,1:-1]-\
            (v[2:,1:-1]-v[:-2,1:-1])/dy[1:-1,1:-1])
        self.Domain.tau12[1:-1,1:-1]=mu*((v[1:-1,2:]-v[1:-1,:-2])/dx[1:-1,1:-1]+\
            (u[2:,1:-1]-u[:-2,1:-1])/dy[1:-1,1:-1])
        self.Domain.tau22[1:-1,1:-1]=2.0/3*mu*(2*(v[2:,1:-1]-v[:-2,1:-1])/dy[1:-1,1:-1]-\
            (u[1:-1,2:]-u[1:-1,:-2])/dx[1:-1,1:-1])
        # Forward/backward differences for boundary values (corners not calculated)
        dx/=2.0
        dy/=2.0
        self.Domain.tau11[0,1:-1] =2.0/3*mu*(2*(u[0,2:]-u[0,1:-1])/dx[0,1:-1]-\
            (v[1,1:-1]-v[0,1:-1])/dy[0,1:-1])
        self.Domain.tau11[-1,1:-1]=2.0/3*mu*(2*(u[-1,2:]-u[-1,1:-1])/dx[-1,1:-1]-\
            (v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1])
        self.Domain.tau11[1:-1,0] =2.0/3*mu*(2*(u[1:-1,1]-u[1:-1,0])/dx[1:-1,0]-\
            (v[2:,0]-v[1:-1,0])/dy[1:-1,0])
        self.Domain.tau11[1:-1,-1]=2.0/3*mu*(2*(u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1]-\
            (v[2:,-1]-v[1:-1,-1])/dy[1:-1,-1])

        self.Domain.tau12[0,1:-1] =mu*((v[0,2:]-v[0,1:-1])/dx[0,1:-1]+\
            (u[1,1:-1]-u[0,1:-1])/dy[0,1:-1])
        self.Domain.tau12[-1,1:-1]=mu*((v[-1,2:]-v[-1,1:-1])/dx[-1,1:-1]+\
            (u[-1,1:-1]-u[-2,1:-1])/dy[-1,1:-1])
        self.Domain.tau12[1:-1,0] =mu*((v[1:-1,1]-v[1:-1,0])/dx[1:-1,0]+\
            (u[2:,0]-u[1:-1,0])/dy[1:-1,0])
        self.Domain.tau12[1:-1,-1]=mu*((v[1:-1,-1]-v[1:-1,-2])/dx[1:-1,-1]+\
            (u[2:,-1]-u[1:-1,-1])/dy[1:-1,-1])

        self.Domain.tau22[0,1:-1] =2.0/3*mu*(2*(v[1,1:-1]-v[0,1:-1])/dy[0,1:-1]-\
            (u[0,2:]-u[0,1:-1])/dx[0,1:-1])
        self.Domain.tau22[-1,1:-1]=2.0/3*mu*(2*(v[-1,1:-1]-v[-2,1:-1])/dy[-1,1:-1]-\
            (u[-1,2:]-u[-1,1:-1])/dx[-1,1:-1])
        self.Domain.tau22[1:-1,0] =2.0/3*mu*(2*(v[2:,0]-v[1:-1,0])/dy[1:-1,0]-\
            (u[1:-1,1]-u[1:-1,0])/dx[1:-1,0])
        self.Domain.tau22[1:-1,-1]=2.0/3*mu*(2*(v[2:,-1]-v[1:-1,-1])/dy[1:-1,-1]-\
            (u[1:-1,-1]-u[1:-1,-2])/dx[1:-1,-1])
    
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
        dx*=2
        dy*=2
        qx=numpy.empty_like(T)
        qy=numpy.empty_like(T)
        k=self.Domain.k
        # Central difference
        qx[:,1:-1]=-k*(T[:,2:]-T[:,:-2])/dx[:,1:-1]
        qy[1:-1,:]=-k*(T[2:,:]-T[:-2,:])/dy[1:-1,:]
        # Forward/backward difference for boundaries
        dx/=2
        dy/=2
        qx[:,0] =-k*(T[:,1]-T[:,0])/dx[:,0]
        qx[:,-1]=-k*(T[:,-1]-T[:,-2])/dx[:,-1]
        
        qy[0,:] =-k*(T[1,:]-T[0,:])/dy[0,:]
        qy[-1,:]=-k*(T[-1,:]-T[-2,:])/dy[-1,:]
        
        return self.compute_Flux(1.0,qx,qy,dx,dy)
    
    # Bondary condition handler (not including periodic BCs)
    def Apply_BCs(self, rho, rhou, rhov, rhoE, u, v, p, T):
        # Start with wall BCs (implied 0 gradients and no slip)
        
        # Left face
        if self.BCs['bc_type_left']=='wall':
            rho[:,0]=rho[:,1]
            p[:,0]  =p[:,1]
            u[:,0]  =0
            v[:,0]  =0
            if self.BCs['bc_left_T']=='zero_grad':
                T[:,0]  =T[:,1]
            else:
                T[:,0]  =self.BCs['bc_left_T']
            
        elif self.BCs['bc_type_left']!='periodic':
            if self.BCs['bc_left_rho']=='zero_grad':
                rho[:,0]=rho[:,1]
            else:
                rho[:,0]=self.BCs['bc_left_rho']
            if self.BCs['bc_left_p']=='zero_grad':
                p[:,0]  =p[:,1]
            else:
                p[:,0]  =self.BCs['bc_left_p']
            if self.BCs['bc_left_u']=='zero_grad':
                u[:,0]  =u[:,1]
            else:
                u[:,0]  =self.BCs['bc_left_u']
            if self.BCs['bc_left_v']=='zero_grad':
                v[:,0]  =v[:,1]
            else:
                v[:,0]  =self.BCs['bc_left_v']    
            if self.BCs['bc_left_T']=='zero_grad':
                T[:,0]  =T[:,1]
            else:
                T[:,0]  =self.BCs['bc_left_T']
        
        # Right face
        if self.BCs['bc_type_right']=='wall':
            rho[:,-1]=rho[:,-2]
            p[:,-1]  =p[:,-2]
            u[:,-1]  =0
            v[:,-1]  =0
            if self.BCs['bc_right_T']=='zero_grad':
                T[:,-1]  =T[:,-2]
            else:
                T[:,-1]  =self.BCs['bc_right_T']
        
        elif self.BCs['bc_type_right']!='periodic':
            if self.BCs['bc_right_rho']=='zero_grad':
                rho[:,-1]=rho[:,-2]
            else:
                rho[:,-1]=self.BCs['bc_right_rho']
            if self.BCs['bc_right_p']=='zero_grad':
                p[:,-1]  =p[:,-2]  
            else:
                p[:,-1]  =self.BCs['bc_right_p']
            if self.BCs['bc_right_u']=='zero_grad':
                u[:,-1]  =u[:,-2]  
            else:
                u[:,-1]  =self.BCs['bc_right_u']
            if self.BCs['bc_right_v']=='zero_grad':
                v[:,-1]  =v[:,-2]  
            else:
                v[:,-1]  =self.BCs['bc_right_v']
            if self.BCs['bc_right_T']=='zero_grad':
                T[:,-1]  =T[:,-2]  
            else:
                T[:,-1]  =self.BCs['bc_right_T']
        
        # South face
        if self.BCs['bc_type_south']=='wall':
            rho[0,:]=rho[1,:]
            p[0,:]  =p[1,:]
            u[0,:]  =0
            v[0,:]  =0
            if self.BCs['bc_south_T']=='zero_grad':
                T[0,:]  =T[1,:]
            else:
                T[0,:]  =self.BCs['bc_south_T']
            
        elif self.BCs['bc_type_south']!='periodic':
            if self.BCs['bc_south_rho']=='zero_grad':
                rho[0,:]=rho[1,:]
            else:
                rho[0,:]=self.BCs['bc_south_rho']
            if self.BCs['bc_south_p']=='zero_grad':
                p[0,:]  =p[1,:]  
            else:
                p[0,:]  =self.BCs['bc_south_p']
            if self.BCs['bc_south_u']=='zero_grad':
                u[0,:]  =u[1,:]  
            else:
                u[0,:]  =self.BCs['bc_south_u']
            if self.BCs['bc_south_v']=='zero_grad':
                v[0,:]  =v[1,:]  
            else:
                v[0,:]  =self.BCs['bc_south_v']
            if self.BCs['bc_south_T']=='zero_grad':
                T[0,:]  =T[1,:]  
            else:
                T[0,:]  =self.BCs['bc_south_T']                
            
        # North face
        if self.BCs['bc_type_north']=='wall':
            rho[-1,:]=rho[-2,:]
            p[-1,:]  =p[-2,:]
            u[-1,:]  =0
            v[-1,:]  =0
            if self.BCs['bc_north_T']=='zero_grad':
                T[-1,:]  =T[-2,:]
            else:
                T[-1,:]  =self.BCs['bc_north_T']
            
        elif self.BCs['bc_type_north']!='periodic':
            if self.BCs['bc_north_rho']=='zero_grad':
                rho[-1,:]=rho[-2,:]
            else:
                rho[-1,:]=self.BCs['bc_north_rho']
            if self.BCs['bc_north_p']=='zero_grad':
                p[-1,:]  =p[-2,:]  
            else:
                p[-1,:]  =self.BCs['bc_north_p']
            if self.BCs['bc_north_u']=='zero_grad':
                u[-1,:]  =u[-2,:]  
            else:
                u[-1,:]  =self.BCs['bc_north_u']
            if self.BCs['bc_north_v']=='zero_grad':
                v[-1,:]  =v[-2,:]  
            else:
                v[-1,:]  =self.BCs['bc_north_v']
            if self.BCs['bc_north_T']=='zero_grad':
                T[-1,:]  =T[-2,:]  
            else:
                T[-1,:]  =self.BCs['bc_north_T']
        
        # Conservative values at boundaries
        if self.BCs['bc_type_left']!='periodic':
            rhou[:,0]=rho[:,0]*u[:,0]
            rhov[:,0]=rho[:,0]*v[:,0]
            rhoE[:,0]=rho[:,0]*\
                (0.5*(u[:,0]**2+v[:,0]**2)+\
                 self.Domain.Cv*T[:,0])
        
        if self.BCs['bc_type_right']!='periodic':
            rhou[:,-1]=rho[:,-1]*u[:,-1]
            rhov[:,-1]=rho[:,-1]*v[:,-1]
            rhoE[:,-1]=rho[:,-1]*\
                (0.5*(u[:,-1]**2+v[:,-1]**2)+\
                 self.Domain.Cv*T[:,-1])
        
        if self.BCs['bc_type_south']!='periodic':
            rhou[0,:]=rho[0,:]*u[0,:]
            rhov[0,:]=rho[0,:]*v[0,:]
            rhoE[0,:]=rho[0,:]*\
                (0.5*(u[0,:]**2+v[0,:]**2)+\
                 self.Domain.Cv*T[0,:])
        
        if self.BCs['bc_type_north']!='periodic':
            rhou[-1,:]=rho[-1,:]*u[-1,:]
            rhov[-1,:]=rho[-1,:]*v[-1,:]
            rhoE[-1,:]=rho[-1,:]*\
                (0.5*(u[-1,:]**2+v[-1,:]**2)+\
                 self.Domain.Cv*T[-1,:])
        
   
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
        drhodt=numpy.zeros_like(rho_c)
        drhoudt=numpy.zeros_like(rhou_c)
        drhovdt=numpy.zeros_like(rhov_c)
        drhoEdt=numpy.zeros_like(rhoE_c)
        
        dt=self.getdt()
        
        if self.time_scheme=='Explicit':
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
                return 1 # Scheme not recognized; abort solve
            rk_coeff = RK_info.rk_coeff
            rk_substep_fraction = RK_info.rk_substep_fraction

            drhodt =[0]*Nstep
            drhoudt=[0]*Nstep
            drhovdt=[0]*Nstep
            drhoEdt=[0]*Nstep
            
        
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
    
            # x-momentum (flux, pressure, shear stress)
            drhoudt[step] =-self.compute_Flux(rhou_c, u, v, self.dx, self.dy)
            drhoudt[step]-=self.compute_Flux(1.0, p, numpy.zeros_like(v), self.dx, self.dy)
    #        drhoudt[1:-1,1:-1]-=(self.Domain.p[1:-1,2:]-self.Domain.p[1:-1,:-2])/(2*self.dx[1:-1,1:-1])
            drhoudt[step]+=self.compute_Flux(1.0, self.Domain.tau11, self.Domain.tau12, self.dx, self.dy)
    
            # y-momentum (flux, pressure, shear stress)
            drhovdt[step] =-self.compute_Flux(rhov_c, u, v, self.dx, self.dy)
            drhovdt[step]-=self.compute_Flux(1.0, numpy.zeros_like(u), p, self.dx, self.dy)
    #        drhovdt[1:-1,1:-1]-=(self.Domain.p[2:,1:-1]-self.Domain.p[:-2,1:-1])/(2*self.dy[1:-1,1:-1])
            drhovdt[step]+=self.compute_Flux(1.0, self.Domain.tau12, self.Domain.tau22, self.dx, self.dy)
            
            # Energy (flux, pressure-work, shear-work, conduction)
            drhoEdt[step] =-self.compute_Flux(rhoE_c, u, v, self.dx, self.dy)
            drhoEdt[step]-=self.compute_Flux(p, u, v, self.dx, self.dy)
            drhoEdt[step]+=self.Source_CSWork(u, v, self.dx, self.dy)
            drhoEdt[step]-=self.Source_Cond(T, self.dx, self.dy)

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
            self.Apply_BCs(rho_c, rhou_c, rhov_c, rhoE_c, u, v, p, T)
            
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
                       self.Domain.rhoE, u, v, p, T)
        
        ###################################################################
        # Output data to file?????
        ###################################################################
        
        
        
        return 0
        
##### FORMER EXPLICIT-TRANS CONDUCTION SOLVER INCLUDING BC CALCULATION #################
    def SolveExpTrans(self):
        Tc=numpy.empty_like(self.T)
        BC=self.BCs # Copy global variables into local ones for easy calling
        dx=self.dx
        dy=self.dy
        Fo=self.Fo
        k=self.Domain.mat_prop['k']
        BC1x,BC1y='T','T'# BC types at corner 1
        BC2x,BC2y='T','T'# BC types at corner 2
        BC3x,BC3y='T','T'# BC types at corner 3
        BC4x,BC4y='T','T'# BC types at corner 4
        # Assign temperature BCs if applicable
        for i in range(len(BC['BCx1'])/3):
            if BC['BCx1'][3*i]=='T':
                st=BC['BCx1'][2+3*i][0]
                en=BC['BCx1'][2+3*i][1]
                self.T[st:en,0]=BC['BCx1'][1+3*i]
                if len(BC['BCx1'])/3-i==1:
                    self.T[-1,0]=BC['BCx1'][-2]
        for i in range(len(BC['BCx2'])/3):
            if BC['BCx2'][3*i]=='T':
                st=BC['BCx2'][2+3*i][0]
                en=BC['BCx2'][2+3*i][1]
                self.T[st:en,-1]=BC['BCx2'][1+3*i]
                if len(BC['BCx2'])/3-i==1:
                    self.T[-1,-1]=BC['BCx2'][-2]
        for i in range(len(BC['BCy1'])/3):
            if BC['BCy1'][3*i]=='T':
                st=BC['BCy1'][2+3*i][0]
                en=BC['BCy1'][2+3*i][1]
                self.T[0,st:en]=BC['BCy1'][1+3*i]
                if len(BC['BCy1'])/3-i==1:
                    self.T[0,-1]=BC['BCy1'][-2]
        for i in range(len(BC['BCy2'])/3):
            if BC['BCy2'][3*i]=='T':
                st=BC['BCy2'][2+3*i][0]
                en=BC['BCy2'][2+3*i][1]
                self.T[-1,st:en]=BC['BCy2'][1+3*i]
                if len(BC['BCy2'])/3-i==1:
                    self.T[-1,-1]=BC['BCy2'][-2]
        
        # Solve temperatures for each time step
        for j in range(self.Nt):
            if (j+1)%100==0:
                print 'Time step %i'%(j+1)
            Tc=self.T.copy()
            self.T[1:-1,1:-1]=2*Fo/(dx[1:,1:]+dx[:-1,:-1])*(Tc[1:-1,:-2]/dx[:-1,:-1]+Tc[1:-1,2:]/dx[1:,1:])\
            +2*Fo/(dy[1:,1:]+dy[:-1,:-1])*(Tc[2:,1:-1]/dy[1:,1:]+Tc[:-2,1:-1]/dy[:-1,:-1])\
            +(1-4*Fo/(dx[1:,1:]+dx[:-1,:-1])/(dy[1:,1:]+dy[:-1,:-1])\
              *((dy[1:,1:]+dy[:-1,:-1])/2/dx[:-1,:-1]+(dy[1:,1:]+dy[:-1,:-1])/2/dx[1:,1:]\
                +(dx[1:,1:]+dx[:-1,:-1])/2/dy[1:,1:]+(dx[1:,1:]+dx[:-1,:-1])/2/dy[:-1,:-1]))*Tc[1:-1,1:-1]
            
            # Apply flux/conv BC at smallest x
            for i in range(len(BC['BCx1'])/3):
                if BC['BCx1'][3*i]=='F' or BC['BCx1'][3*i]=='C':
                    st=BC['BCx1'][2+3*i][0]
                    en=BC['BCx1'][2+3*i][1]
                    if BC['BCx1'][3*i]=='F':
                        q=BC['BCx1'][1+3*i]
                        Bi=0
                        if i==0:
                            BC1x='F'
                    else:
                        q=BC['BCx1'][1+3*i][0]*BC['BCx1'][1+3*i][1] # h*Tinf
                        Bi=BC['BCx1'][1+3*i][0]/k
                        print('Convective BC %i'%q)
                        if i==0:
                            BC1x='C'
                    
                    self.T[st:en,0]=2*Fo*q/k/dx[st:en+1,0]+2*Fo/dx[st:en+1,0]**2*Tc[st:en,1]\
                        +2*Fo/(dy[st:en+1,0]+dy[st-1:en,0])\
                        *(Tc[st-1:en-1,0]/dy[st-1:en,0]+Tc[st+1:en+1,0]/dy[st:en+1,0])\
                        +(1-2*Fo/dx[st:en+1,0]**2-2*Fo/(dy[st:en+1,0]+dy[st-1:en,0])\
                        *(1/dy[st-1:en,0]+1/dy[st:en+1,0])-2*Fo*Bi/dx[st:en+1,0])*Tc[st:en,0]
                    if len(BC['BCx1'])/3-i==1:
                        self.T[-2,0]=2*Fo*q/k/dx[-1,0]+2*Fo/dx[-1,0]**2*Tc[-2,1]\
                            +2*Fo/(dy[-1,0]+dy[-2,0])\
                            *(Tc[-3,0]/dy[-2,0]+Tc[-1,0]/dy[-1,0])\
                            +(1-2*Fo/dx[-1,0]**2-2*Fo/(dy[-1,0]+dy[-2,0])\
                            *(1/dy[-2,0]+1/dy[-1,0])-2*Fo*Bi/dx[-1,0])*Tc[-2,0]
                        if Bi==0:
                            BC4x='F'
                        else:
                            BC4x='C'
            
            # Apply flux/conv BC at largest x
            for i in range(len(BC['BCx2'])/3):
                if BC['BCx2'][3*i]=='F' or BC['BCx2'][3*i]=='C':
                    st=BC['BCx2'][2+3*i][0]
                    en=BC['BCx2'][2+3*i][1]
                    if BC['BCx2'][3*i]=='F':
                        q=BC['BCx2'][1+3*i]
                        Bi=0
                        if i==0:
                            BC2x='F'
                    else:
                        q=BC['BCx2'][1+3*i][0]*BC['BCx2'][1+3*i][1] # h*Tinf
                        Bi=BC['BCx2'][1+3*i][0]/k
                        if i==0:
                            BC2x='C'
                    self.T[st:en,-1]=2*Fo*q/k/dx[st:en+1,-1]+2*Fo/dx[st:en+1,-1]**2*Tc[st:en,-2]\
                        +2*Fo/(dy[st:en+1,-1]+dy[st-1:en,-1])\
                        *(Tc[st-1:en-1,-1]/dy[st-1:en,-1]+Tc[st+1:en+1,-1]/dy[st:en+1,-1])\
                        +(1-2*Fo/dx[st:en+1,-1]**2-2*Fo/(dy[st:en+1,-1]+dy[st-1:en,-1])\
                        *(1/dy[st-1:en,-1]+1/dy[st:en+1,-1])-2*Fo*Bi/dx[st:en+1,-1])*Tc[st:en,-1]

                    if len(BC['BCx2'])/3-i==1:
                        self.T[-2,-1]=2*Fo*q/k/dx[-1,-1]+2*Fo/dx[-1,-1]**2*Tc[-2,-2]\
                            +2*Fo/(dy[-1,-1]+dy[-2,-1])\
                            *(Tc[-3,-1]/dy[-2,-1]+Tc[-1,-1]/dy[-1,-1])\
                            +(1-2*Fo/dx[-1,0]**2-2*Fo/(dy[-1,0]+dy[-2,-1])\
                            *(1/dy[-2,-1]+1/dy[-1,-1])-2*Fo*Bi/dx[-1,-1])*Tc[-2,-1]
                        if Bi==0:
                            BC3x='F'
                        else:
                            BC3x='C'
                        
            # Apply flux/conv BC at smallest y
            for i in range(len(BC['BCy1'])/3):
                if BC['BCy1'][3*i]=='F' or BC['BCy1'][3*i]=='C':
                    st=BC['BCy1'][2+3*i][0]
                    en=BC['BCy1'][2+3*i][1]
                    if BC['BCy1'][3*i]=='F':
                        q=BC['BCy1'][1+3*i]
                        Bi=0
                        if i==0:
                            BC1y='F'
                    else:
                        q=BC['BCy1'][1+3*i][0]*BC['BCy1'][1+3*i][1] # h*Tinf
                        Bi=BC['BCy1'][1+3*i][0]/k
                        if i==0:
                            BC1y='C'
                    self.T[0,st:en]=2*Fo*q/k/dy[0,st:en+1]+2*Fo/dy[0,st:en+1]**2*Tc[1,st:en]\
                        +2*Fo/(dx[0,st:en+1]+dx[0,st-1:en])\
                        *(Tc[0,st-1:en-1]/dx[0,st-1:en]+Tc[0,st+1:en+1]/dx[0,st:en+1])\
                        +(1-2*Fo/dy[0,st:en+1]**2-2*Fo/(dx[0,st:en+1]+dx[0,st-1:en])\
                        *(1/dx[0,st-1:en]+1/dx[0,st:en+1])-2*Fo*Bi/dy[0,st:en+1])*Tc[0,st:en]

                    if len(BC['BCy1'])/3-i==1:
                        self.T[0,-2]=2*Fo*q/k/dy[0,-1]+2*Fo/dy[0,-1]**2*Tc[1,-2]\
                            +2*Fo/(dx[0,-1]+dx[0,-2])\
                            *(Tc[0,-3]/dx[0,-2]+Tc[0,-1]/dx[0,-1])\
                            +(1-2*Fo/dy[0,-1]**2-2*Fo/(dx[0,-1]+dx[0,-2])\
                            *(1/dx[0,-2]+1/dx[0,-1])-2*Fo*Bi/dy[0,-1])*Tc[0,-2]
                        if Bi==0:
                            BC2y='F'
                        else:
                            BC2y='C'
            ## Apply flux/conv BC at largest y (CHANGED dx and dy array ops)
            for i in range(len(BC['BCy2'])/3):
                if BC['BCy2'][3*i]=='F' or BC['BCy2'][3*i]=='C':
                    st=BC['BCy2'][2+3*i][0]
                    en=BC['BCy2'][2+3*i][1]
                    if BC['BCy2'][3*i]=='F':
                        q=BC['BCy2'][1+3*i]
                        Bi=0
                        if i==0:
                            BC4y='F'
                    else:
                        q=BC['BCy2'][1+3*i][0]*BC['BCy2'][1+3*i][1] # h*Tinf
                        Bi=BC['BCy2'][1+3*i][0]/k
                        if i==0:
                            BC4y='C'
                    self.T[-1,st:en]=2*Fo*q/k/dy[-1,st:en+1]+2*Fo/dy[-1,st:en+1]**2*Tc[-2,st:en]\
                        +2*Fo/(dx[-1,st:en+1]+dx[-1,st-1:en])\
                        *(Tc[-1,st-1:en-1]/dx[-1,st-1:en]+Tc[-1,st+1:en+1]/dx[-1,st:en+1])\
                        +(1-2*Fo/dy[-1,st:en+1]**2-2*Fo/(dx[-1,st:en+1]+dx[-1,st-1:en])\
                        *(1/dx[-1,st-1:en]+1/dx[-1,st:en+1])-2*Fo*Bi/dy[-1,st:en+1])*Tc[-1,st:en]

                    if len(BC['BCy2'])/3-i==1:
                        self.T[-1,-2]=2*Fo*q/k/dy[-1,-1]+2*Fo/dy[-1,-1]**2*Tc[-2,-2]\
                            +2*Fo/(dx[-1,-1]+dx[-1,-2])\
                            *(Tc[-1,-3]/dx[-1,-2]+Tc[-1,-1]/dx[-1,-1])\
                            +(1-2*Fo/dy[-1,-1]**2-2*Fo/(dx[-1,-1]+dx[-1,-2])\
                            *(1/dx[-1,-2]+1/dx[-1,-1])-2*Fo*Bi/dy[-1,-1])*Tc[-1,-2]
                        if Bi==0:
                            BC3y='F'
                        else:
                            BC3y='C'

            # Corner treatments
            if (BC1x!='T' and BC1y!='T'):
                if BC1x=='F':
                    qx=BC['BCx1'][1] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx1'][1][0]*BC['BCx1'][1][1] # h*Tinf for conv
                    Bix=BC['BCx1'][1][0]/k
                if BC1y=='F':
                    qy=BC['BCy1'][1] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy1'][1][0]*BC['BCy1'][1][1] # h*Tinf for conv
                    Biy=BC['BCy1'][1][0]/k
                
                self.T[0,0]=2*Fo*qx/k/dx[0,0]+2*Fo*qy/k/dy[0,0]\
                    +2*Fo/dx[0,0]**2*Tc[0,1]+2*Fo/dy[0,0]**2*Tc[1,0]\
                    +(1-2*Fo*(1/dx[0,0]**2+1/dy[0,0]**2)\
                      -2*Fo*(Bix/dx[0,0]+Biy/dy[0,0]))*Tc[0,0]

            if (BC2x!='T' and BC2y!='T'):
                if BC2x=='F':
                    qx=BC['BCx2'][1] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx2'][1][0]*BC['BCx2'][1][1] # h*Tinf for conv
                    Bix=BC['BCx2'][1][0]/k
                if BC2y=='F':
                    qy=BC['BCy1'][-2] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy1'][-2][0]*BC['BCy1'][-2][1] # h*Tinf for conv
                    Biy=BC['BCy1'][-2][0]/k
                
                self.T[0,-1]=2*Fo*qx/k/dx[0,-1]+2*Fo*qy/k/dy[0,-1]\
                    +2*Fo/dx[0,-1]**2*Tc[0,-2]+2*Fo/dy[0,-1]**2*Tc[1,-1]\
                    +(1-2*Fo*(1/dx[0,-1]**2+1/dy[0,-1]**2)\
                      -2*Fo*(Bix/dx[0,-1]+Biy/dy[0,-1]))*Tc[0,-1]                
            
            if (BC3x!='T' and BC3y!='T'):
                if BC3x=='F':
                    qx=BC['BCx2'][-2] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx2'][-2][0]*BC['BCx2'][-2][1] # h*Tinf for conv
                    Bix=BC['BCx2'][-2][0]/k
                if BC3y=='F':
                    qy=BC['BCy2'][-2] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy2'][-2][0]*BC['BCy2'][-2][1] # h*Tinf for conv
                    Biy=BC['BCy2'][-2][0]/k
                
                self.T[-1,-1]=2*Fo*qx/k/dx[-1,-1]+2*Fo*qy/k/dy[-1,-1]\
                    +2*Fo/dx[-1,-1]**2*Tc[-1,-2]+2*Fo/dy[-1,-1]**2*Tc[-2,-1]\
                    +(1-2*Fo*(1/dx[-1,-1]**2+1/dy[-1,-1]**2)\
                      -2*Fo*(Bix/dx[-1,-1]+Biy/dy[-1,-1]))*Tc[-1,-1]                 
            
            if (BC4x!='T' and BC4y!='T'):
                if BC4x=='F':
                    qx=BC['BCx1'][-2] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx1'][-2][0]*BC['BCx1'][-2][1] # h*Tinf for conv
                    Bix=BC['BCx1'][-2][0]/k
                if BC4y=='F':
                    qy=BC['BCy2'][1] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy2'][1][0]*BC['BCy2'][1][1] # h*Tinf for conv
                    Biy=BC['BCy2'][1][0]/k
                
                self.T[-1,0]=2*Fo*qx/k/dx[-1,0]+2*Fo*qy/k/dy[-1,0]\
                    +2*Fo/dx[-1,0]**2*Tc[-1,1]+2*Fo/dy[-1,0]**2*Tc[-2,0]\
                    +(1-2*Fo*(1/dx[-1,0]**2+1/dy[-1,0]**2)\
                      -2*Fo*(Bix/dx[-1,0]+Biy/dy[-1,0]))*Tc[-1,0]                 

######### FORMER CONDUCTION STEADY STATE SOLVER INCLUDING BC CALCULATOR #######
    def SolveSS(self):
        Tc=numpy.empty_like(self.T)
        count=0
        BC=self.BCs # Copy global variables into local ones for easy calling
        dx=self.dx
        dy=self.dy
        k=self.Domain.mat_prop['k']
        BC1x,BC1y='T','T'# BC types at corner 1
        BC2x,BC2y='T','T'# BC types at corner 2
        BC3x,BC3y='T','T'# BC types at corner 3
        BC4x,BC4y='T','T'# BC types at corner 4
        
        # Assign temperature BCs if applicable
        for i in range(len(BC['BCx1'])/3):
            if BC['BCx1'][3*i]=='T':
                st=BC['BCx1'][2+3*i][0]
                en=BC['BCx1'][2+3*i][1]
                self.T[st:en,0]=BC['BCx1'][1+3*i]
                if len(BC['BCx1'])/3-i==1:
                    self.T[-1,0]=BC['BCx1'][-2]
        for i in range(len(BC['BCx2'])/3):
            if BC['BCx2'][3*i]=='T':
                st=BC['BCx2'][2+3*i][0]
                en=BC['BCx2'][2+3*i][1]
                self.T[st:en,-1]=BC['BCx2'][1+3*i]
                if len(BC['BCx2'])/3-i==1:
                    self.T[-1,-1]=BC['BCx2'][-2]
        for i in range(len(BC['BCy1'])/3):
            if BC['BCy1'][3*i]=='T':
                st=BC['BCy1'][2+3*i][0]
                en=BC['BCy1'][2+3*i][1]
                self.T[0,st:en]=BC['BCy1'][1+3*i]
                if len(BC['BCy1'])/3-i==1:
                    self.T[0,-1]=BC['BCy1'][-2]
        for i in range(len(BC['BCy2'])/3):
            if BC['BCy2'][3*i]=='T':
                st=BC['BCy2'][2+3*i][0]
                en=BC['BCy2'][2+3*i][1]
                self.T[-1,st:en]=BC['BCy2'][1+3*i]
                if len(BC['BCy2'])/3-i==1:
                    self.T[-1,-1]=BC['BCy2'][-2]

        print 'Residuals:'
        while count<self.maxCount:
            Tc=self.T.copy()
            self.T[1:-1,1:-1]=(Tc[:-2,1:-1]/self.dy[:-1,:-1]+Tc[2:,1:-1]/self.dy[1:,1:]\
            +Tc[1:-1,:-2]/self.dx[:-1,:-1]+Tc[1:-1,2:]/self.dx[1:,1:])\
            /(1/self.dx[1:,1:]+1/self.dx[:-1,:-1]+1/self.dy[1:,:-1]+1/self.dy[:-1,:-1])
            
            # Apply flux/conv BC at smallest x
            for i in range(len(BC['BCx1'])/3):
                if BC['BCx1'][3*i]=='F' or BC['BCx1'][3*i]=='C':
                    st=BC['BCx1'][2+3*i][0]
                    en=BC['BCx1'][2+3*i][1]
                    if BC['BCx1'][3*i]=='F':
                        q=BC['BCx1'][1+3*i]
                        Bi=0
                        if i==0:
                            BC1x='F'
                    else:
                        q=BC['BCx1'][1+3*i][0]*BC['BCx1'][1+3*i][1] # h*Tinf
                        Bi=BC['BCx1'][1+3*i][0]/k
                        if i==0:
                            BC1x='C'
#                    self.T[st:en,0]=(2*q*dy[1,1]**2*dx[1,1]/k+2*dy[1,1]**2*self.T[st:en,1]\
#                         +dx[1,1]**2*(self.T[st-1:en-1,0]+self.T[st+1:en+1,0]))\
#                         /(2*dy[1,1]**2+2*dx[1,1]**2) ################# equal spacings
                    
                    self.T[st:en,0]=((dy[st:en+1,0]+dy[st-1:en,0])*(q/k+self.T[st:en,1]/dx[st-1:en,0])\
                             +dx[st-1:en,0]*(self.T[st-1:en-1,0]/dy[st-1:en,0]\
                                +self.T[st+1:en+1,0]/dy[st:en+1,0]))\
                             /((dy[st:en+1,0]+dy[st-1:en,0])*(1/dx[st-1:en,0]+Bi)\
                               +dx[st-1:en,0]*(1/dy[st-1:en,0]+1/dy[st:en+1,0]))
                    if len(BC['BCx1'])/3-i==1:
#                        self.T[-2,0]=(2*q*dy[1,1]**2*dx[1,1]/k+2*dy[1,1]**2*self.T[-2,1]\
#                             +dx[1,1]**2*(self.T[-3,0]+self.T[-1,0]))\
#                             /(2*dy[1,1]**2+2*dx[1,1]**2)##################### equal spacings
                        
                        self.T[-2,0]=((dy[-1,0]+dy[-2,0])*(q/k+self.T[-2,1]/dx[-2,0])\
                                 +dx[-2,0]*(self.T[-3,0]/dy[-2,0]\
                                    +self.T[-1,0]/dy[-1,0]))\
                                 /((dy[-1,0]+dy[-2,0])*(1/dx[-2,0]+Bi)\
                                   +dx[-2,0]*(1/dy[-2,0]+1/dy[-1,0]))
                        if Bi==0:
                            BC4x='F'
                        else:
                            BC4x='C'
            
            # Apply flux/conv BC at largest x
            for i in range(len(BC['BCx2'])/3):
                if BC['BCx2'][3*i]=='F' or BC['BCx2'][3*i]=='C':
                    st=BC['BCx2'][2+3*i][0]
                    en=BC['BCx2'][2+3*i][1]
                    if BC['BCx2'][3*i]=='F':
                        q=BC['BCx2'][1+3*i]
                        Bi=0
                        if i==0:
                            BC2x='F'
                    else:
                        q=BC['BCx2'][1+3*i][0]*BC['BCx2'][1+3*i][1] # h*Tinf
                        Bi=BC['BCx2'][1+3*i][0]/k
                        if i==0:
                            BC2x='C'
#                    self.T[st:en,-1]=(2*q*dy[1,1]**2*dx[1,1]/k+2*dy[1,1]**2*self.T[st:en,-2]\
#                         +dx[1,1]**2*(self.T[st-1:en-1,-1]+self.T[st+1:en+1,-1]))\
#                         /(2*dy[1,1]**2+2*dx[1,1]**2) ################# equal spacings
                    
                    self.T[st:en,-1]=((dy[st:en+1,-1]+dy[st-1:en,-1])*(q/k+self.T[st:en,-2]/dx[st-1:en,-1])\
                             +dx[st-1:en,0]*(self.T[st-1:en-1,-1]/dy[st-1:en,-1]\
                                +self.T[st+1:en+1,-1]/dy[st:en+1,-1]))\
                             /((dy[st:en+1,-1]+dy[st-1:en,-1])*(1/dx[st-1:en,-1]+Bi)\
                               +dx[st-1:en,-1]*(1/dy[st-1:en,-1]+1/dy[st:en+1,-1]))

                    if len(BC['BCx2'])/3-i==1:
#                        self.T[-2,-1]=(2*q*dy[1,1]**2*dx[1,1]/k+2*dy[1,1]**2*self.T[-2,-2]\
#                         +dx[1,1]**2*(self.T[-3,-1]+self.T[-1,-1]))\
#                         /(2*dy[1,1]**2+2*dx[1,1]**2) ########################### equal spacings
                        
                        self.T[-2,-1]=((dy[-1,-1]+dy[-2,-1])*(q/k+self.T[-2,-2]/dx[-2,-1])\
                                         +dx[-2,0]*(self.T[-3,-1]/dy[-2,-1]\
                                            +self.T[-1,-1]/dy[-1,-1]))\
                                         /((dy[-1,-1]+dy[-2,-1])*(1/dx[-2,-1]+Bi)\
                                           +dx[-2,-1]*(1/dy[-1,-1]+1/dy[-2,-1]))
                        if Bi==0:
                            BC3x='F'
                        else:
                            BC3x='C'
                        
            # Apply flux/conv BC at smallest y
            for i in range(len(BC['BCy1'])/3):
                if BC['BCy1'][3*i]=='F' or BC['BCy1'][3*i]=='C':
                    st=BC['BCy1'][2+3*i][0]
                    en=BC['BCy1'][2+3*i][1]
                    if BC['BCy1'][3*i]=='F':
                        q=BC['BCy1'][1+3*i]
                        Bi=0
                        if i==0:
                            BC1y='F'
                    else:
                        q=BC['BCy1'][1+3*i][0]*BC['BCy1'][1+3*i][1] # h*Tinf
                        Bi=BC['BCy1'][1+3*i][0]/k
                        if i==0:
                            BC1y='C'
#                    self.T[0,st:en]=(2*q*dx[1,1]**2*dy[1,1]/k+2*dx[1,1]**2*self.T[1,st:en]\
#                         +dy[1,1]**2*(self.T[0,st-1:en-1]+self.T[0,st+1:en+1]))\
#                         /(2*dx[1,1]**2+2*dy[1,1]**2)############# equal spacing
                    
                    self.T[0,st:en]=((dx[0,st:en+1]+dx[0,st-1:en])*(q/k+self.T[1,st:en]/dy[0,st-1:en])\
                             +dy[0,st-1:en]*(self.T[0,st-1:en-1]/dx[0,st-1:en]\
                                +self.T[0,st+1:en+1]/dx[0,st:en+1]))\
                             /((dx[0,st:en+1]+dx[0,st-1:en])*(1/dy[0,st-1:en]+Bi)\
                               +dy[0,st-1:en]*(1/dx[0,st-1:en]+1/dx[0,st:en+1]))

                    if len(BC['BCy1'])/3-i==1:
#                        self.T[0,-2]=(2*q*dx[1,1]**2*dy[1,1]/k+2*dx[1,1]**2*self.T[1,-2]\
#                         +dy[1,1]**2*(self.T[0,-3]+self.T[0,-1]))\
#                         /(2*dx[1,1]**2+2*dy[1,1]**2)##################### equal spacing
                        
                        self.T[0,-2]=((dx[0,-1]+dx[0,-2])*(q/k+self.T[1,-2]/dy[0,-2])\
                                 +dy[0,-2]*(self.T[0,-3]/dx[0,-2]\
                                    +self.T[0,-1]/dx[0,-1]))\
                                 /((dx[0,-1]+dx[0,-2])*(1/dy[0,-2]+Bi)\
                                   +dy[0,-2]*(1/dx[0,-2]+1/dx[0,-1]))
                        if Bi==0:
                            BC2y='F'
                        else:
                            BC2y='C'
                        
            # Apply flux/conv BC at largest y
            for i in range(len(BC['BCy2'])/3):
                if BC['BCy2'][3*i]=='F' or BC['BCy2'][3*i]=='C':
                    st=BC['BCy2'][2+3*i][0]
                    en=BC['BCy2'][2+3*i][1]
                    if BC['BCy2'][3*i]=='F':
                        q=BC['BCy2'][1+3*i]
                        Bi=0
                        if i==0:
                            BC4y='F'
                    else:
                        q=BC['BCy2'][1+3*i][0]*BC['BCy2'][1+3*i][1] # h*Tinf
                        Bi=BC['BCy2'][1+3*i][0]/k
                        if i==0:
                            BC4y='C'
#                    self.T[-1,st:en]=(2*q*dx[1,1]**2*dy[1,1]/k+2*dx[1,1]**2*self.T[-2,st:en]\
#                         +dy[1,1]**2*(self.T[-1,st-1:en-1]+self.T[-1,st+1:en+1]))\
#                         /(2*dx[1,1]**2+2*dy[1,1]**2)##################### equal spacing
#                    
                    self.T[-1,st:en]=((dx[-1,st:en+1]+dx[-1,st-1:en])*(q/k+self.T[-2,st:en]/dy[-1,st-1:en])\
                             +dy[-1,st-1:en]*(self.T[-1,st-1:en-1]/dx[-1,st-1:en]\
                                +self.T[-1,st+1:en+1]/dx[-1,st:en+1]))\
                             /((dx[-1,st:en+1]+dx[-1,st-1:en])*(1/dy[-1,st-1:en]+Bi)\
                               +dy[-1,st-1:en]*(1/dx[-1,st-1:en]+1/dx[-1,st:en+1]))

                    if len(BC['BCy2'])/3-i==1:
#                        self.T[-1,-2]=(2*q*dx[1,1]**2*dy[1,1]/k+2*dx[1,1]**2*self.T[-2,-2]\
#                         +dy[1,1]**2*(self.T[-1,-3]+self.T[-1,-1]))\
#                         /(2*dx[1,1]**2+2*dy[1,1]**2)###################### equal spacing
                        
                        self.T[-1,-2]=((dx[-1,-1]+dx[-1,-2])*(q/k+self.T[-2,-2]/dy[-1,-2])\
                                 +dy[-1,-2]*(self.T[-1,-3]/dx[-1,-2]\
                                    +self.T[-1,-1]/dx[-1,-1]))\
                                 /((dx[-1,-1]+dx[-1,-2])*(1/dy[-1,-2]+Bi)\
                                   +dy[-1,-2]*(1/dx[-1,-2]+1/dx[-1,-1]))
                        if Bi==0:
                            BC3y='F'
                        else:
                            BC3y='C'
                        
            # Corner treatments
            if (BC1x!='T' and BC1y!='T'):
                if BC1x=='F':
                    qx=BC['BCx1'][1] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx1'][1][0]*BC['BCx1'][1][1] # h*Tinf for conv
                    Bix=BC['BCx1'][1][0]*dx[0,0]/k
                if BC1y=='F':
                    qy=BC['BCy1'][1] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy1'][1][0]*BC['BCy1'][1][1] # h*Tinf for conv
                    Biy=BC['BCy1'][1][0]*dy[0,0]/k
                
                self.T[0,0]=(dx[0,0]**2*dy[0,0]/k*qy+dy[0,0]**2*dx[0,0]/k*qx \
                    +dy[0,0]**2*self.T[0,1]+dx[0,0]**2*self.T[1,0])\
                      /(dy[0,0]**2+dx[0,0]**2+dx[0,0]**2*Biy+dy[0,0]**2*Bix)

            if (BC2x!='T' and BC2y!='T'):
                if BC2x=='F':
                    qx=BC['BCx2'][1] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx2'][1][0]*BC['BCx2'][1][1] # h*Tinf for conv
                    Bix=BC['BCx2'][1][0]*dx[0,-1]/k
                if BC2y=='F':
                    qy=BC['BCy1'][-2] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy1'][-2][0]*BC['BCy1'][-2][1] # h*Tinf for conv
                    Biy=BC['BCy1'][-2][0]*dy[0,-1]/k
                
                self.T[0,-1]=(dx[0,-1]**2*dy[0,-1]/k*qy+dy[0,-1]**2*dx[0,-1]/k*qx \
                    +dy[0,-1]**2*self.T[0,-2]+dx[0,-1]**2*self.T[1,-1])\
                      /(dy[0,-1]**2+dx[0,-1]**2+dx[0,-1]**2*Biy+dy[0,-1]**2*Bix)
            
            if (BC3x!='T' and BC3y!='T'):
                if BC3x=='F':
                    qx=BC['BCx2'][-2] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx2'][-2][0]*BC['BCx2'][-2][1] # h*Tinf for conv
                    Bix=BC['BCx2'][-2][0]*dx[-1,-1]/k
                if BC3y=='F':
                    qy=BC['BCy2'][-2] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy2'][-2][0]*BC['BCy2'][-2][1] # h*Tinf for conv
                    Biy=BC['BCy2'][-2][0]*dy[-1,-1]/k
                
                self.T[-1,-1]=(dx[-1,-1]**2*dy[-1,-1]/k*qy+dy[-1,-1]**2*dx[-1,-1]/k*qx \
                    +dy[0,-1]**2*self.T[-1,-2]+dx[-1,-1]**2*self.T[-2,-1])\
                      /(dy[-1,-1]**2+dx[-1,-1]**2+dx[-1,-1]**2*Biy+dy[-1,-1]**2*Bix)
            
            if (BC4x!='T' and BC4y!='T'):
                if BC4x=='F':
                    qx=BC['BCx1'][-2] # flux value on x
                    Bix=0
                else:
                    qx=BC['BCx1'][-2][0]*BC['BCx1'][-2][1] # h*Tinf for conv
                    Bix=BC['BCx1'][-2][0]*dx[-1,0]/k
                if BC4y=='F':
                    qy=BC['BCy2'][1] # flux value on y
                    Biy=0
                else:
                    qy=BC['BCy2'][1][0]*BC['BCy2'][1][1] # h*Tinf for conv
                    Biy=BC['BCy2'][1][0]*dy[-1,0]/k
                
                self.T[-1,0]=(dx[-1,0]**2*dy[-1,0]/k*qy+dy[-1,0]**2*dx[-1,0]/k*qx \
                    +dy[0,0]**2*self.T[-1,1]+dx[-1,0]**2*self.T[-2,0])\
                      /(dy[-1,0]**2+dx[-1,0]**2+dx[-1,0]**2*Biy+dy[-1,0]**2*Bix)

            if self.CheckConv(Tc,self.T):
                break            