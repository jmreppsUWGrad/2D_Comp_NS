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

#import numpy as np
#import string as st

class BCs():
    def __init__(self, settings, BC_dict, dx, dy):
        self.BCs=BC_dict
        self.dx,self.dy=dx,dy
        self.R=settings['R']
        self.gamma=settings['gamma']
        self.Cv=self.R/(self.gamma-1)
    
    # Apply BCs to stress and heat transfer (viscous BCs)
    def Visc_BCs(self, tau11, tau12, tau22, qx, qy):
        # Start with wall BCs
        
        # Left face
        if self.BCs['bc_type_left']=='slip_wall':
           tau12[:,0]=0
            
        elif self.BCs['bc_type_left']=='outlet':
            tau12[:,0]=tau12[:,1]
            qx[:,0]=qx[:,1]
        
        # Periodic boundary       
#        else:
#            rho[:,0] =rho[:,-1]
#            rhou[:,0]=rhou[:,-1]
#            rhov[:,0]=rhov[:,-1]
#            rhoE[:,0]=rhoE[:,-1]
        
        # Right face
        if self.BCs['bc_type_right']=='slip_wall':
            tau12[:,-1]=0
        
        elif self.BCs['bc_type_right']=='outlet':
            tau12[:,-1]=tau12[:,-2]
            qx[:,-1]=qx[:,-2]
        
        # Periodic boundary 
#        else:
#            rho[:,-1] =rho[:,0]
#            rhou[:,-1]=rhou[:,0]
#            rhov[:,-1]=rhov[:,0]
#            rhoE[:,-1]=rhoE[:,0]
            
        # South face
        if self.BCs['bc_type_south']=='slip_wall':
            tau12[0,:]=0
        
        elif self.BCs['bc_type_south']=='outlet':
            tau12[0,:]=tau12[1,:]
            qy[0,:]=qy[1,:]
        
        # Periodic boundary       
#        else:
#            rho[0,:] =rho[-1,:]
#            rhou[0,:]=rhou[-1,:]
#            rhov[0,:]=rhov[-1,:]
#            rhoE[0,:]=rhoE[-1,:]
            
        # North face
        if self.BCs['bc_type_north']=='slip_wall':
            tau12[-1,:]=0
        
        elif self.BCs['bc_type_north']=='outlet':
            tau12[-1,:]=tau12[-2,:]
            qy[-1,:]=qy[-2,:]
        
        # Periodic boundary       
#        else:
#            rho[-1,:] =rho[0,:]
#            rhou[-1,:]=rhou[0,:]
#            rhov[-1,:]=rhov[0,:]
#            rhoE[-1,:]=rhoE[0,:]
    
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
        else:
            rho[:,0] =rho[:,-1]
            rhou[:,0]=rhou[:,-1]
            rhov[:,0]=rhov[:,-1]
            rhoE[:,0]=rhoE[:,-1]
        
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
        
        else:
            rho[:,-1] =rho[:,0]
            rhou[:,-1]=rhou[:,0]
            rhov[:,-1]=rhov[:,0]
            rhoE[:,-1]=rhoE[:,0]
            
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
        else:
            rho[0,:] =rho[-1,:]
            rhou[0,:]=rhou[-1,:]
            rhov[0,:]=rhov[-1,:]
            rhoE[0,:]=rhoE[-1,:]
            
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
        else:
            rho[-1,:] =rho[0,:]
            rhou[-1,:]=rhou[0,:]
            rhov[-1,:]=rhov[0,:]
            rhoE[-1,:]=rhoE[0,:]