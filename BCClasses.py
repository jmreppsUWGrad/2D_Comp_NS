# -*- coding: utf-8 -*-
"""
######################################################
#       2D Compressible Navier-Stokes Solver         #
#              Created by J. Mark Epps               #
#          Part of Masters Thesis at UW 2018-2020    #
######################################################

This file contains functions to do boundary conditions:
    -
    
Features:
    -

Desired:
    -
    -
    
"""

import numpy as np
import string as st

class BCs():
    def __init__(self, settings, BC_dict, dx, dy):
        self.BCs=BC_dict
        self.dx,self.dy=dx,dy
        self.R=settings['R']
        self.gamma=settings['gamma']
        self.mu=settings['mu']
        self.Cv=self.R/(self.gamma-1)
        self.hx,self.hy=dx,dy
        self.k=settings['k']
    
    # Interpolation function (copied from solver object)
    def interpolate(self, k1, k2, func):
        if func=='Linear':
            return 0.5*k1+0.5*k2
        else:
            return 2*k1*k2/(k1+k2)
    
    # Apply BCs to stress and heat transfer (viscous BCs)
    # u,v are velocities if doing viscous stresses, u is temp if qx/qy
    # Boundary tau or q are implied 0 if nothing done
    def Visc_BCs(self, tau11, tau12, tau21, tau22, qx, qy, u, v, isWork):
        if isWork:
            uw=u
            vw=v
        else:
            uw=np.ones_like(u)
            vw=np.ones_like(v)
        # Start with wall BCs
        
        # Left face
#        if st.find(self.BCs['bc_type_left'], 'adiabatic')>=0:
#            qx[:,0]=0
            
        if st.find(self.BCs['bc_type_left'], 'outlet')>=0:
            tau21[:,0]=0
            qx[:,0]=0
        
        if st.find(self.BCs['bc_type_left'], 'inlet')>=0:
            tau11[:,0]=0
        
        # Right face
#        if st.find(self.BCs['bc_type_right'], 'adiabatic')>=0:
#            qx[:,-1]=0
            
        if st.find(self.BCs['bc_type_right'], 'outlet')>=0:
            tau21[:,-1]=0
            qx[:,-1]=0
        
        if st.find(self.BCs['bc_type_right'], 'inlet')>=0:
            tau11[:,-1]=0
        
        # Periodic BCs on left/right faces
        if st.find(self.BCs['bc_type_left'], 'periodic')>=0\
            or st.find(self.BCs['bc_type_right'], 'periodic')>=0:
            # Heat flux
            qx[:,0]   += (self.k*(u[:,-1]-u[:,0])/self.dx[:,0])/self.hx[:,0]
            qx[:,-1]  += (self.k*(u[:,0]-u[:,-1])/self.dx[:,-1])/self.hx[:,-1]
            
            ############## tau 11 flux (d/dx)
            # Left face at x=0
            tau11[:,0] -=4.0/3*self.mu/self.hx[:,0]*(u[:,0]-u[:,-1])/self.dx[:,0]\
                *self.interpolate(uw[:,0],uw[:,-1], 'Linear')
                # y gradients
            tau11[1:-1,0] -=2.0/3*self.mu/self.hx[1:-1,0]*(\
                0.5*(v[2:,-1]-v[:-2,-1])/(self.dy[1:-1,-1]+self.dy[:-2,-1])+\
                0.5*(v[2:,0]-v[:-2,0])/(self.dy[1:-1,0]+self.dy[:-2,0]))\
                *self.interpolate(uw[1:-1,0],uw[1:-1,-1], 'Linear')
            tau11[0,0] -=2.0/3*self.mu/self.hx[0,0]*(\
                0.5*(v[1,-1]-v[0,-1])/self.dy[0,-1]+\
                0.5*(v[1,0]-v[0,0])/self.dy[0,0])\
                *self.interpolate(uw[0,0],uw[0,-1], 'Linear')
            tau11[-1,0] -=2.0/3*self.mu/self.hx[-1,0]*(\
                0.5*(v[-1,-1]-v[-2,-1])/self.dy[-1,-1]+\
                0.5*(v[-1,0]-v[-2,0])/self.dy[-1,0])\
                *self.interpolate(uw[-1,0],uw[-1,-1], 'Linear')
            # Right face at x=L
            tau11[:,-1]+=4.0/3*self.mu/self.hx[:,-1]*(u[:,0]-u[:,-1])/self.dx[:,-1]\
                *self.interpolate(uw[:,0],uw[:,-1], 'Linear')
                # y gradients
            tau11[1:-1,-1] +=2.0/3*self.mu/self.hx[1:-1,-1]*(\
                0.5*(v[2:,-1]-v[:-2,-1])/(self.dy[1:-1,-1]+self.dy[:-2,-1])+\
                0.5*(v[2:,0]-v[:-2,0])/(self.dy[1:-1,0]+self.dy[:-2,0]))\
                *self.interpolate(uw[1:-1,0],uw[1:-1,-1], 'Linear')
            tau11[0,-1] +=2.0/3*self.mu/self.hx[0,-1]*(\
                0.5*(v[1,-1]-v[0,-1])/self.dy[0,-1]+\
                0.5*(v[1,0]-v[0,0])/self.dy[0,0])\
                *self.interpolate(uw[0,0],uw[0,-1], 'Linear')
            tau11[-1,-1] +=2.0/3*self.mu/self.hx[-1,-1]*(\
                0.5*(v[-1,-1]-v[-2,-1])/self.dy[-1,-1]+\
                0.5*(v[-1,0]-v[-2,0])/self.dy[-1,0])\
                *self.interpolate(uw[-1,0],uw[-1,-1], 'Linear')
            
            ############## tau 21 flux (d/dx)
            # Left face at x=0
            tau21[:,0] -=self.mu/self.hx[:,0]*(v[:,0]-v[:,-1])/self.dx[:,0]\
                *self.interpolate(vw[:,0],vw[:,-1], 'Linear')
                # y gradients
            tau21[1:-1,0] -=self.mu/self.hx[1:-1,0]*(\
                0.5*(u[2:,-1]-u[:-2,-1])/(self.dy[1:-1,0]+self.dy[:-2,-1])+\
                0.5*(u[2:,0]-u[:-2,0])/(self.dy[1:-1,0]+self.dy[:-2,0]))\
                *self.interpolate(vw[1:-1,0],vw[1:-1,-1], 'Linear')
            tau21[0,0] -=self.mu/self.hx[0,0]*(\
                0.5*(u[1,-1]-u[0,-1])/self.dy[0,-1]+\
                0.5*(u[1,0]-u[0,0])/self.dy[0,0])\
                *self.interpolate(vw[0,0],vw[0,-1], 'Linear')
            tau21[-1,1:] -=self.mu/self.hx[-1,1:]*(\
                0.5*(u[-1,-1]-u[-2,-1])/self.dy[-1,-1]+\
                0.5*(u[-1,0]-u[-2,0])/self.dy[-1,0])\
                *self.interpolate(vw[-1,0],vw[-1,-1], 'Linear')
            # Right face at x=L
            tau21[:,-1]+=self.mu/self.hx[:,-1]*(v[:,0]-v[:,-1])/self.dx[:,-1]\
                *self.interpolate(vw[:,0],vw[:,-1], 'Linear')
                # y gradients
            tau21[1:-1,-1] +=self.mu/self.hx[1:-1,-1]*(\
                0.5*(u[2:,-1]-u[:-2,-1])/(self.dy[1:-1,-1]+self.dy[:-2,-1])+\
                0.5*(u[2:,0]-u[:-2,0])/(self.dy[1:-1,0]+self.dy[:-2,0]))\
                *self.interpolate(vw[1:-1,0],vw[1:-1,-1], 'Linear')
            tau21[0,-1] +=self.mu/self.hx[0,-1]*(\
                0.5*(u[1,-1]-u[0,-1])/self.dy[0,-1]+\
                0.5*(u[1,0]-u[0,0])/self.dy[0,0])\
                *self.interpolate(vw[0,0],vw[0,-1], 'Linear')
            tau21[-1,-1] +=self.mu/self.hx[-1,-1]*(\
                0.5*(u[-1,-1]-u[-2,-1])/self.dy[-1,-1]+\
                0.5*(u[-1,0]-u[-2,0])/self.dy[-1,0])\
                *self.interpolate(vw[-1,0],vw[-1,-1], 'Linear')
            
            
        # South face
#        if st.find(self.BCs['bc_type_south'], 'adiabatic')>=0:
#            qy[0,:]=0
        
        if st.find(self.BCs['bc_type_south'], 'outlet')>=0:
            tau12[0,:]=0
            qy[0,:]=0
        
        if st.find(self.BCs['bc_type_south'], 'inlet')>=0:
            tau22[0,:]=0
            
        # North face
#        if st.find(self.BCs['bc_type_north'], 'adiabatic')>=0:
#            qy[-1,:]=0
        
        if st.find(self.BCs['bc_type_north'], 'outlet')>=0:
            tau12[-1,:]=0
            qy[-1,:]=0
            
        if st.find(self.BCs['bc_type_north'], 'inlet')>=0:
            tau22[-1,:]=0
            
        
        # Periodic boundary       
        if st.find(self.BCs['bc_type_north'], 'periodic')>=0\
            or st.find(self.BCs['bc_type_south'], 'periodic')>=0:
            ############## tau 12 flux (d/dy)
            # Bottom face
            tau12[0,:] -=self.mu/self.hy[0,:]*(u[0,:]-u[-1,:])/self.dy[-1,:]\
                *self.interpolate(uw[0,:],uw[-1,:], 'Linear')
                # x gradients
            tau12[0,1:-1] -=self.mu/self.hy[0,1:-1]*(\
                0.5*(v[-1,2:]-v[-1,:-2])/(self.dx[-1,1:-1]+self.dx[-1,:-2])+\
                0.5*(v[0,2:]-v[0,:-2])/(self.dx[0,1:-1]+self.dx[0,:-2]))\
                *self.interpolate(uw[0,1:-1],uw[-1,1:-1], 'Linear')
            tau12[0,0] -=self.mu/self.hy[0,0]*(\
                0.5*(v[-1,1]-v[-1,0])/self.dx[-1,0]+\
                0.5*(v[0,1]-v[0,0])/self.dx[0,0])\
                *self.interpolate(uw[0,0],uw[-1,0], 'Linear')
            tau12[0,-1] -=self.mu/self.hy[0,-1]*(\
                0.5*(v[-1,-1]-v[-1,-2])/self.dx[-1,-1]+\
                0.5*(v[0,-1]-v[0,-2])/self.dx[0,-1])\
                *self.interpolate(uw[0,-1],uw[-1,-1], 'Linear')
            # Top face
            tau12[-1,:]+=self.mu/self.hy[-1,:]*(u[0,:]-u[-1,:])/self.dy[-1,:]\
                *self.interpolate(uw[0,:],uw[-1,:], 'Linear')
                # x gradients
            tau12[-1,1:-1] +=self.mu/self.hy[-1,1:-1]*(\
                0.5*(v[-1,2:]-v[-1,:-2])/(self.dx[-1,1:-1]+self.dx[-1,:-2])+\
                0.5*(v[0,2:]-v[0,:-2])/(self.dx[0,1:-1]+self.dx[0,:-2]))\
                *self.interpolate(uw[0,1:-1],uw[-1,1:-1], 'Linear')
            tau12[-1,0] +=self.mu/self.hy[-1,0]*(\
                0.5*(v[-1,1]-v[-1,0])/self.dx[-1,0]+\
                0.5*(v[0,1]-v[0,0])/self.dx[0,0])\
                *self.interpolate(uw[0,0],uw[-1,0], 'Linear')
            tau12[-1,-1] +=self.mu/self.hy[-1,-1]*(\
                0.5*(v[-1,-1]-v[-1,-2])/self.dx[-1,-1]+\
                0.5*(v[0,-1]-v[0,-2])/self.dx[0,-1])\
                *self.interpolate(uw[0,-1],uw[-1,-1], 'Linear')
            
            ############## tau 22 flux (d/dy)
            # Bottom face
            tau22[0,:] -=4.0/3*self.mu/self.hy[0,:]*(v[0,:]-v[-1,:])/self.dy[-1,:]\
                *self.interpolate(vw[0,:],vw[-1,:], 'Linear')
                # x gradients
            tau22[0,1:-1] -=2.0/3*self.mu/self.hy[0,1:-1]*(\
                0.5*(u[-1,2:]-u[-1,:-2])/(self.dx[-1,1:-1]+self.dx[-1,:-2])+\
                0.5*(u[0,2:]-u[0,:-2])/(self.dx[0,1:-1]+self.dx[0,:-2]))\
                *self.interpolate(vw[0,1:-1],vw[-1,1:-1], 'Linear')
            tau22[0,0] -=2.0/3*self.mu/self.hy[0,0]*(\
                0.5*(u[-1,1]-u[-1,0])/self.dx[-1,0]+\
                0.5*(u[0,1]-u[0,0])/self.dx[0,0])\
                *self.interpolate(vw[0,0],vw[-1,0], 'Linear')
            tau22[0,-1] -=2.0/3*self.mu/self.hy[0,-1]*(\
                0.5*(u[-1,-1]-u[-1,-2])/self.dx[-1,-1]+\
                0.5*(u[0,-1]-u[0,-2])/self.dx[0,-1])\
                *self.interpolate(vw[0,-1],vw[-1,-1], 'Linear')
            
            # Top face
            tau22[-1,:]+=4.0/3*self.mu/self.hy[-1,:]*(v[0,:]-v[-1,:])/self.dy[-1,:]\
                *self.interpolate(vw[0,:],vw[-1,:], 'Linear')
                # x gradients
            tau22[-1,1:-1] +=2.0/3*self.mu/self.hy[-1,1:-1]*(\
                0.5*(u[-1,2:]-u[-1,:-2])/(self.dx[-1,1:-1]+self.dx[-1,:-2])+\
                0.5*(u[0,2:]-u[0,:-2])/(self.dx[0,1:-1]+self.dx[0,:-2]))\
                *self.interpolate(vw[0,1:-1],vw[-1,1:-1], 'Linear')
            tau22[-1,0] +=2.0/3*self.mu/self.hy[-1,0]*(\
                0.5*(u[-1,1]-u[-1,0])/self.dx[-1,0]+\
                0.5*(u[0,1]-u[0,0])/self.dx[0,0])\
                *self.interpolate(vw[0,0],vw[-1,0], 'Linear')
            tau22[-1,-1] +=2.0/3*self.mu/self.hy[-1,-1]*(\
                0.5*(u[-1,-1]-u[-1,-2])/self.dx[-1,-1]+\
                0.5*(u[0,-1]-u[0,-2])/self.dx[0,-1])\
                *self.interpolate(vw[0,-1],vw[-1,-1], 'Linear')
            
    # Apply conservative BCs (inviscid BCs)
    def Apply_BCs(self, rho, rhou, rhov, rhoE, u, v, p, T):
        # Start with wall BCs
        
        # Left face
        if self.BCs['bc_type_left']=='wall':
#            print 'Left: wall'
            p[:,0]  =p[:,1]
            rhou[:,0]  =0
            rhov[:,0]  =0
            T[:,0]  =self.BCs['bc_left_T']
            rho[:,0]=p[:,0]/(self.R*T[:,0])
#            rhoE[:,0]=rho[:,0]*self.Cv*T[:,0]
            rhoE[:,0]=p[:,0]/(self.gamma-1)
        
        elif self.BCs['bc_type_left']=='slip_wall':
            rhou[:,0]  =0
            p[:,0]  =p[:,1]
            T[:,0]  =self.BCs['bc_left_T']
            rho[:,0]=p[:,0]/(self.R*T[:,0])
#            rhoE[:,0]=rho[:,0]*(0.5*(v[:,0]**2)+self.Cv*T[:,0])
            rhoE[:,0]=p[:,0]/(self.gamma-1)
            
        elif self.BCs['bc_type_left']=='inlet':
            p[:,0]  =self.BCs['bc_left_p']
#            pt      =self.BCs['bc_left_p']
            u[:,0]  =self.BCs['bc_left_u']
            v[:,0]  =self.BCs['bc_left_v']
            T[:,0]  =self.BCs['bc_left_T']
            
#            u[:,0]=np.sqrt(2*self.gamma*self.R*T[:,0]/(self.gamma-1)\
#                 *((p[:,0]/pt)**(self.gamma/(self.gamma-1))-1))
#            p[:,0]=pt*(1+(self.gamma-1)/2*u[:,0]/(self.gamma*self.R*T[:,0]))\
#                 **((self.gamma-1)/self.gamma)

#            p[:,0]=rho[:,0]*self.R*T[:,0]
            rho[:,0]=p[:,0]/(self.R*T[:,0])
            rhou[:,0]=rho[:,0]*u[:,0]
            rhov[:,0]=rho[:,0]*v[:,0]
#            rhoE[:,0]=p[:,0]/(self.gamma-1)+rho[:,0]*0.5*(u[:,0]**2+v[:,0]**2)
            rhoE[:,0]=rho[:,0]*(0.5*(u[:,0]**2+v[:,0]**2)+self.Cv*T[:,0])
                
        elif self.BCs['bc_type_left']=='outlet':
            p[:,0]=self.BCs['bc_left_p']
            rhoE[:,0]=p[:,0]/(self.gamma-1)+rho[:,0]*0.5*(u[:,0]**2+v[:,0]**2)
        
        # Periodic boundary       
#        else:
#            rho[:,0] +=1.0/(2*self.hx[:,0])*self.interpolate(rho[:,0],rho[:,-1],'Linear')\
#                *self.interpolate(u[:,0], u[:,-1],'Linear')#\
##                -LLF*lam*(u[:,1:]-u[:,:-1])
#            rhou[:,0]=rhou[:,-1]
#            rhov[:,0]=rhov[:,-1]
#            rhoE[:,0]=rhoE[:,-1]
        
        # Right face
        if self.BCs['bc_type_right']=='wall':
#            print 'Right: wall'
            p[:,-1]  =p[:,-2]
            rhou[:,-1]  =0
            rhov[:,-1]  =0
#            if type(self.BCs['bc_right_T']) is tuple:
#                T[:,-1]  =T[:,-2]+self.BCs['bc_right_T'][1]*dx[:,-1]
            T[:,-1]  =self.BCs['bc_right_T']
            rho[:,-1]=p[:,-1]/(self.R*T[:,-1])
#            rhoE[:,-1]=rho[:,-1]*self.Cv*T[:,-1]
            rhoE[:,-1]=p[:,-1]/(self.gamma-1)
            
        elif self.BCs['bc_type_right']=='slip_wall':
            p[:,-1]  =p[:,-2]
            rhou[:,-1]  =0
#            if type(self.BCs['bc_right_T']) is tuple:
#                T[:,-1]  =T[:,-2]+self.BCs['bc_right_T'][1]*dx[:,-1]
            T[:,-1]  =self.BCs['bc_right_T']
            rho[:,-1]=p[:,-1]/(self.R*T[:,-1])
#            rhoE[:,-1]=rho[:,-1]*(0.5*(v[:,-1]**2)+self.Cv*T[:,-1])
            rhoE[:,-1]=p[:,-1]/(self.gamma-1)
        
        elif self.BCs['bc_type_right']=='inlet':
            u[:,-1]  =self.BCs['bc_right_u']
            v[:,-1]  =self.BCs['bc_right_v']
            T[:,-1]  =self.BCs['bc_right_T']
            p[:,-1]  =self.BCs['bc_right_p']
            
            rho[:,-1]=p[:,-1]/self.R/T[:,-1]
            rhou[:,-1]=rho[:,-1]*u[:,-1]
            rhov[:,-1]=rho[:,-1]*v[:,-1]
#            rhoE[:,-1]=p[:,-1]/(self.gamma-1)+rho[:,-1]*0.5*(u[:,-1]**2+v[:,-1]**2)
            rhoE[:,-1]=rho[:,-1]*(0.5*(u[:,-1]**2+v[:,-1]**2)+self.Cv*T[:,-1])
                
        elif self.BCs['bc_type_right']=='outlet':
            p[:,-1]=self.BCs['bc_right_p']
            rhoE[:,-1]=p[:,-1]/(self.gamma-1)+rho[:,-1]*0.5*(u[:,-1]**2+v[:,-1]**2)
        
#        else:
#            rho[:,-1] =rho[:,0]
#            rhou[:,-1]=rhou[:,0]
#            rhov[:,-1]=rhov[:,0]
#            rhoE[:,-1]=rhoE[:,0]
            
        # South face
        if self.BCs['bc_type_south']=='wall':
#            print 'South: wall'
            p[0,:]  =p[1,:]
            rhou[0,:]  =0
            rhov[0,:]  =0
            T[0,:]  =self.BCs['bc_south_T']
            rho[0,:]=p[0,:]/(self.R*T[0,:])
#            rhoE[0,:]=rho[0,:]*self.Cv*T[0,:]
            rhoE[0,:]=p[0,:]/(self.gamma-1)
            
        elif self.BCs['bc_type_south']=='slip_wall':
            p[0,:]  =p[1,:]
            rhov[0,:]  =0
            T[0,:]  =self.BCs['bc_south_T']
            rho[0,:]=p[0,:]/(self.R*T[0,:])
#            rhoE[0,:]=rho[0,:]*(0.5*(u[0,:]**2)+self.Cv*T[0,:])
            rhoE[0,:]=p[0,:]/(self.gamma-1)
        
        elif self.BCs['bc_type_south']=='inlet':
            u[0,:]  =self.BCs['bc_south_u']
            v[0,:]  =self.BCs['bc_south_v']
            T[0,:]  =self.BCs['bc_south_T']
            p[0,:]  =self.BCs['bc_south_p']
            
            rho[0,:]=p[0,:]/self.R/T[0,:]
            rhou[0,:]=rho[0,:]*u[0,:]
            rhov[0,:]=rho[0,:]*v[0,:]
#            rhoE[0,:]=p[0,:]/(self.gamma-1)+rho[0,:]*0.5*(u[0,:]**2+v[0,:]**2)
            rhoE[0,:]=rho[0,:]*(0.5*(u[0,:]**2+v[0,:]**2)+self.Cv*T[0,:])
                
        elif self.BCs['bc_type_south']=='outlet':
            p[0,:]=self.BCs['bc_south_p']
            rhoE[0,:]=p[0,:]/(self.gamma-1)+rho[0,:]*0.5*(u[0,:]**2+v[0,:]**2)
        
        # Periodic boundary       
#        else:
#            rho[0,:] =rho[-1,:]
#            rhou[0,:]=rhou[-1,:]
#            rhov[0,:]=rhov[-1,:]
#            rhoE[0,:]=rhoE[-1,:]
            
        # North face
        if self.BCs['bc_type_north']=='wall':
#            print 'North: wall'
            p[-1,:]  =p[-2,:]
            rhou[-1,:]  =0
            rhov[-1,:]  =0
            T[-1,:]  =self.BCs['bc_north_T']
            rho[-1,:]=p[-1,:]/(self.R*T[-1,:])
#            rhoE[-1,:]=rho[-1,:]*self.Cv*T[-1,:]
            rhoE[-1,:]=p[-1,:]/(self.gamma-1)
            
        elif self.BCs['bc_type_north']=='slip_wall':
            p[-1,:]  =p[-2,:]
            rhov[-1,:]  =0
            T[-1,:]  =self.BCs['bc_north_T']
            rho[-1,:]=p[-1,:]/(self.R*T[-1,:])
#            rhoE[-1,:]=rho[-1,:]*(0.5*(u[-1,:]**2)+self.Cv*T[-1,:])
            rhoE[-1,:]=p[-1,:]/(self.gamma-1)
        
        elif self.BCs['bc_type_north']=='inlet':
            u[-1,:]  =self.BCs['bc_north_u']
            v[-1,:]  =self.BCs['bc_north_v']
            T[-1,:]  =self.BCs['bc_north_T']
            p[-1,:]  =self.BCs['bc_north_p']
            
            rho[-1,:]=p[-1,:]/self.R/T[-1,:]
            rhou[-1,:]=rho[-1,:]*u[-1,:]
            rhov[-1,:]=rho[-1,:]*v[-1,:]
#            rhoE[-1,:]=p[-1,:]/(self.gamma-1)+0.5*rho[-1,:]*(u[-1,:]**2+v[-1,:]**2)
            rhoE[-1,:]=rho[-1,:]*(0.5*(u[-1,:]**2+v[-1,:]**2)+self.Cv*T[-1,:])
                
        elif self.BCs['bc_type_north']=='outlet':
            p[-1,:]=self.BCs['bc_north_p']
            rhoE[-1,:]=p[-1,:]/(self.gamma-1)+0.5*rho[-1,:]*(u[-1,:]**2+v[-1,:]**2)
        
        # Periodic boundary       
#        else:
#            rho[-1,:] =rho[0,:]
#            rhou[-1,:]=rhou[0,:]
#            rhov[-1,:]=rhov[0,:]
#            rhoE[-1,:]=rhoE[0,:]