# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:14:27 2018

This class is a 2D planar domain for solving temp, vel or density.

Requires:
    -length and width of domain
    -number of nodes across length and width
    
Features:
    -Linear biasing one way or two way based on specified smallest element size
    -Creates distance between nodes arrays (dx and dy)

Desired:
    -store CV dimensions (dx and dy) for each point; makes for equal sized
    arrays with temp/vel/press/density

Features:
    -

@author: Joseph
"""
import numpy as np

class TwoDimPlanar:
    def __init__(self, settings):
        
        self.L=settings['Length']
        self.W=settings['Width']
        self.Nx=settings['Nodes_x']
        self.Ny=settings['Nodes_y']
        self.x=np.zeros(self.Nx)
        self.y=np.zeros(self.Ny)
        self.dx=np.zeros(self.Nx) # NOTE: SIZE MADE TO MATCH REST OF ARRAYS (FOR NOW)
        self.dy=np.zeros(self.Ny) # NOTE: SIZE MADE TO MATCH REST OF ARRAYS (FOR NOW)
        self.fluid=settings['Fluid']
        self.k=settings['k']
        self.gamma=settings['gamma']
        self.mu=settings['mu']
        self.R=settings['R']
        
        # Biasing options       
        self.xbias=[settings['bias_type_x'], settings['bias_size_x']]
        self.ybias=[settings['bias_type_y'], settings['bias_size_y']]
        self.isMeshed=False
        
        # Setup variable arrays
        self.rhoE=np.zeros((self.Ny, self.Nx)) # Conservative arrays
        self.rhou=np.zeros((self.Ny,self.Nx))
        self.rhov=np.zeros((self.Ny,self.Nx))
        self.rho=np.zeros((self.Ny,self.Nx)) # Primitive arrays
#        self.T=np.zeros((self.Ny, self.Nx))
#        self.u=np.zeros((self.Ny,self.Nx))
#        self.v=np.zeros((self.Ny,self.Nx))
#        self.p=np.zeros((self.Ny,self.Nx))
        self.tau11=np.zeros((self.Ny, self.Nx)) # Shear stress arrays
        self.tau12=np.zeros((self.Ny, self.Nx))
#        self.tau21=np.zeros((self.Ny, self.Nx))
        self.tau22=np.zeros((self.Ny, self.Nx))
        
        # Other useful calculations (put elsewhere??)
        self.Cv=self.R/(self.gamma-1)
        
    # Discretize domain and save dx and dy
    def mesh(self):
        # Discretize x
        if self.xbias[0]=='OneWayUp':
            smallest=self.xbias[1]
            self.dx[:-1]=np.linspace(2*self.L/(self.Nx-1)-smallest,smallest,self.Nx-1)
            self.dx[-1]=self.dx[-2]
            print 'One way biasing in x: smallest element at x=%2f'%self.L
            print 'Element size range: %2f, %2f'%(smallest, 2*self.L/(self.Nx-1)-smallest)
        elif self.xbias[0]=='OneWayDown':
            smallest=self.xbias[1]
            self.dx[:-1]=np.linspace(smallest,2*self.L/(self.Nx-1)-smallest,self.Nx-1)
            self.dx[-1]=self.dx[-2]
            print 'One way biasing in x: smallest element at x=0'
            print 'Element size range: %2f, %2f'%(smallest, 2*self.L/(self.Nx-1)-smallest)
        elif self.xbias[0]=='TwoWayEnd':
            smallest=self.xbias[1]
            self.dx[:int(self.Nx/2)]=np.linspace(smallest,2*self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):-1]=np.linspace(2*self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            self.dx[-1]=self.dx[-2]
            print 'Two way biasing in x: smallest elements at x=0 and %2f'%self.L
            print 'Element size range: %2f, %2f'%(smallest, 2*self.L/(self.Nx-1)-smallest)
        elif self.xbias[0]=='TwoWayMid':
            smallest=self.xbias[1]
            self.dx[:int(self.Nx/2)]=np.linspace(2*self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):-1]=np.linspace(smallest,2*self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            self.dx[-1]=self.dx[-2]
            print 'Two way biasing in x: smallest elements around x=%2f'%(self.L/2)
            print 'Element size range: %2f, %2f'%(smallest, 2*self.L/(self.Nx-1)-smallest)
        else:
            self.dx[:]=self.L/(self.Nx-1)
            print 'No biasing schemes specified in x'
        
        # Discretize y
        if self.ybias[0]=='OneWayUp':
            smallest=self.ybias[1]
            self.dy[:-1]=np.linspace(2*self.W/(self.Ny-1)-smallest,smallest,self.Ny-1)
            self.dy[-1]=self.dy[-2]
            print 'One way biasing in y: smallest element at y=%2f'%self.W
            print 'Element size range: %2f, %2f'%(smallest, 2*self.W/(self.Ny-1)-smallest)
        elif self.ybias[0]=='OneWayDown':
            smallest=self.ybias[1]
            self.dy[:-1]=np.linspace(smallest,2*self.W/(self.Ny-1)-smallest,self.Ny-1)
            self.dy[-1]=self.dy[-2]
            print 'One way biasing in y: smallest element at y=0'
            print 'Element size range: %2f, %2f'%(smallest, 2*self.W/(self.Ny-1)-smallest)
        elif self.ybias[0]=='TwoWayEnd':
            smallest=self.ybias[1]
            self.dy[:int(self.Ny/2)]=np.linspace(smallest,2*self.W/(self.Ny-1)-smallest,(self.Ny-1)/2)
            self.dy[int(self.Ny/2):-1]=np.linspace(2*self.W/(self.Ny-1)-smallest,smallest,(self.Ny-1)/2)
            self.dy[-1]=self.dy[-2]
            print 'Two way biasing in y: smallest elements at y=0 and %2f'%self.W
            print 'Element size range: %2f, %2f'%(smallest, 2*self.W/(self.Ny-1)-smallest)
        elif self.ybias[0]=='TwoWayMid':
            smallest=self.ybias[1]
            self.dy[:int(self.Ny/2)]=np.linspace(2*self.W/(self.Ny-1)-smallest,smallest,(self.Ny-1)/2)
            self.dy[int(self.Ny/2):-1]=np.linspace(smallest,2*self.W/(self.Ny-1)-smallest,(self.Ny-1)/2)
            self.dy[-1]=self.dy[-2]
            print 'Two way biasing in y: smallest elements around y=%2f'%(self.W/2)
            print 'Element size range: %2f, %2f'%(smallest, 2*self.W/(self.Ny-1)-smallest)
        else:
            self.dy[:]=self.W/(self.Ny-1)
            print 'No biasing schemes specified in y'

        for i in range(self.Nx-1):
            self.x[i+1]=self.x[i]+self.dx[i]
        for i in range(self.Ny-1):
            self.y[i+1]=self.y[i]+self.dy[i]
        self.X,self.Y=np.meshgrid(self.x,self.y)
        
        self.isMeshed=True
    
    # Calculate and return area of faces at each node
    def CV_dim(self):
        hx=np.zeros_like(self.rho)
        hy=np.zeros_like(self.rho)
        dx,dy=np.meshgrid(self.dx, self.dy)
        # y-dimension
        hy[1:-1,:]=0.5*(dy[1:-1,:]+dy[:-2,:])
        hy[0,:]   =0.5*(dy[0,:])
        hy[-1,:]  =0.5*(dy[-1,:])
        # x-dimension
        hx[:,1:-1]=0.5*(dx[:,1:-1]+dx[:,:-2])
        hx[:,0]   =0.5*(dx[:,0])
        hx[:,-1]  =0.5*(dx[:,-1])
        
        return hx, hy
    
    # Return primitive variables from conservative ones
    def primitiveFromConserv(self, rho, rhou, rhov, rhoE):
        u=rhou/rho
        v=rhov/rho
        # Ideal gas law assumed
        T=(rhoE/rho-0.5*(u**2+v**2))/self.Cv
        p=rho*self.R*T
#        p=(rhoE-0.5*rho*(u**2+v**2))*(self.gamma-1)
#        T=p/(rho*self.R)

        return u,v,p,T
    
    # Calculate temperature dependent properties (unsure if this will be the spot)
    def calcProp(self):
        
        self.Cv=self.mat_prop['R']/(self.mat_prop['gamma']-1)
    
    # Check everything before solving
    def IsReadyToSolve(self):
        if self.isMeshed:
            return True
        else:
            return False
