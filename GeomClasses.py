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
import numpy

class OneDimLine:
    def __init__(self, length, num_nodes):
        self.L=1.0*length
        self.Nx=num_nodes
        self.x=numpy.zeros(self.Nx)
        self.dx=numpy.zeros(self.Nx-1)
        self.bias_elem={'OneWayUp': 0, 'OneWayDown': 0, 'TwoWayEnd': 0, 'TwoWayMid': 0}
        # Setup variable arrays
        self.T=numpy.zeros(self.Nx)
        self.mat_prop={'k': 15, 'Cp': 800, 'rho': 8000}
        self.isMeshed=False
        
    def mesh(self):
        if self.bias_elem['OneWayUp']!=0:
            smallest=self.bias_elem['OneWayUp']
            self.dx=numpy.linspace(2*self.L/(self.Nx-1)-smallest,smallest,self.Nx-1)
            print 'One way biasing: smallest element at x=%2f'%self.L
        elif self.bias_elem['OneWayDown']!=0:
            smallest=self.bias_elem['OneWayDown']
            self.dx=numpy.linspace(smallest,2*self.L/(self.Nx-1)-smallest,self.Nx-1)
            print 'One way biasing: smallest element at x=0'
        elif self.bias_elem['TwoWayEnd']!=0:
            smallest=self.bias_elem['TwoWayEnd']
            self.dx[:int(self.Nx/2)]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            print 'Two way biasing: smallest elements at x=0 and %2f'%self.L
        elif self.bias_elem['TwoWayMid']!=0:
            smallest=self.bias_elem['TwoWayMid']
            self.dx[:int(self.Nx/2)]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            print 'Two way biasing: smallest elements around x=%2f'%(self.L/2)
        else:
            self.dx[:]=self.L/(self.Nx-1)
            print 'No biasing schemes specified'
        
        for i in range(self.Nx-1):
            self.x[i+1]=self.x[i]+self.dx[i]
        
        self.isMeshed=True
    
    def final_check(self):
        if self.isMeshed:
            return True
        else:
            return False
                

class TwoDimPlanar:
    def __init__(self, settings):
        
        self.L=settings['Length']
        self.W=settings['Width']
        self.Nx=settings['Nodes_x']
        self.Ny=settings['Nodes_y']
        self.x=numpy.zeros(self.Nx)
        self.y=numpy.zeros(self.Ny)
        self.dx=numpy.zeros(self.Nx) # NOTE: SIZE MADE TO MATCH REST OF ARRAYS (FOR NOW)
        self.dy=numpy.zeros(self.Ny) # NOTE: SIZE MADE TO MATCH REST OF ARRAYS (FOR NOW)
        self.fluid=settings['Fluid']
        self.k=settings['k']
        self.gamma=settings['gamma']
        self.mu=settings['mu']
        self.R=settings['R']
        self.isMeshed=False
        
        # Dictionaries to define biasing (smallest element)
        self.xbias_elem={'OneWayUp': 0, 'OneWayDown': 0, 'TwoWayEnd': 0, 'TwoWayMid': 0}
        self.ybias_elem={'OneWayUp': 0, 'OneWayDown': 0, 'TwoWayEnd': 0, 'TwoWayMid': 0}
        
        # Setup variable arrays
        self.tau11=numpy.zeros((self.Ny, self.Nx)) # Shear stress arrays
        self.tau12=numpy.zeros((self.Ny, self.Nx))
        self.tau22=numpy.zeros((self.Ny, self.Nx))
        self.rhoE=numpy.zeros((self.Ny, self.Nx)) # Conservative arrays
        self.rhou=numpy.zeros((self.Ny,self.Nx))
        self.rhov=numpy.zeros((self.Ny,self.Nx))
        self.rho=numpy.zeros((self.Ny,self.Nx)) # Primitive arrays
#        self.T=numpy.zeros((self.Ny, self.Nx))
#        self.u=numpy.zeros((self.Ny,self.Nx))
#        self.v=numpy.zeros((self.Ny,self.Nx))
#        self.p=numpy.zeros((self.Ny,self.Nx))
        
        # Other useful calculations (put elsewhere??)
        self.Cv=self.R/(self.gamma-1)
        
    # Discretize domain and save dx and dy
    def mesh(self):
        # Discretize x
        if self.xbias_elem['OneWayUp']!=0:
            smallest=self.xbias_elem['OneWayUp']
            self.dx=numpy.linspace(2*self.L/(self.Nx-1)-smallest,smallest,self.Nx-1)
            print 'One way biasing in x: smallest element at x=%2f'%self.L
        elif self.xbias_elem['OneWayDown']!=0:
            smallest=self.xbias_elem['OneWayDown']
            self.dx=numpy.linspace(smallest,2*self.L/(self.Nx-1)-smallest,self.Nx-1)
            print 'One way biasing in x: smallest element at x=0'
        elif self.xbias_elem['TwoWayEnd']!=0:
            smallest=self.xbias_elem['TwoWayEnd']
            self.dx[:int(self.Nx/2)]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            print 'Two way biasing in x: smallest elements at x=0 and %2f'%self.L
        elif self.xbias_elem['TwoWayMid']!=0:
            smallest=self.xbias_elem['TwoWayMid']
            self.dx[:int(self.Nx/2)]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            print 'Two way biasing in x: smallest elements around x=%2f'%(self.L/2)
        else:
            self.dx[:]=self.L/(self.Nx-1)
            print 'No biasing schemes specified in x'
        
        # Discretize y
        if self.ybias_elem['OneWayUp']!=0:
            smallest=self.ybias_elem['OneWayUp']
            self.dy=numpy.linspace(2*self.W/(self.Ny-1)-smallest,smallest,self.Ny-1)
            print 'One way biasing in y: smallest element at y=%2f'%self.W
        elif self.ybias_elem['OneWayDown']!=0:
            smallest=self.ybias_elem['OneWayDown']
            self.dy=numpy.linspace(smallest,2*self.W/(self.Ny-1)-smallest,self.Ny-1)
            print 'One way biasing in y: smallest element at y=0'
        elif self.ybias_elem['TwoWayEnd']!=0:
            smallest=self.ybias_elem['TwoWayEnd']
            self.dy[:int(self.Ny/2)]=numpy.linspace(smallest,self.W/(self.Ny-1)-smallest,(self.Ny-1)/2)
            self.dy[int(self.Ny/2):]=numpy.linspace(self.W/(self.Ny-1)-smallest,smallest,(self.Ny-1)/2)
            print 'Two way biasing in y: smallest elements at y=0 and %2f'%self.W
        elif self.ybias_elem['TwoWayMid']!=0:
            smallest=self.ybias_elem['TwoWayMid']
            self.dy[:int(self.Ny/2)]=numpy.linspace(self.W/(self.Ny-1)-smallest,smallest,(self.Ny-1)/2)
            self.dy[int(self.Ny/2):]=numpy.linspace(smallest,self.W/(self.Ny-1)-smallest,(self.Ny-1)/2)
            print 'Two way biasing in y: smallest elements around y=%2f'%(self.W/2)
        else:
            self.dy[:]=self.W/(self.Ny-1)
            print 'No biasing schemes specified in y'

        for i in range(self.Nx-1):
            self.x[i+1]=self.x[i]+self.dx[i]
        for i in range(self.Ny-1):
            self.y[i+1]=self.y[i]+self.dy[i]
        self.X,self.Y=numpy.meshgrid(self.x,self.y)
        
        self.isMeshed=True
    
    def primitiveFromConserv(self, rho, rhou, rhov, rhoE):
        u=rhou/rho
        v=rhov/rho
        T=(rhoE/rho-0.5*(u**2+v**2))/self.Cv
        
        # Ideal gas law assumed
        p=rho*self.R*T
        
        return u,v,p,T
    
    # Calculate temperature dependent properties (unsure if this will be the spot)
    def calcTempDepProp(self):
        
        self.Cv=self.mat_prop['R']/(self.mat_prop['gamma']-1)
    
    # Check everything before solving
    def IsReadyToSolve(self):
        if self.isMeshed:
            return True
        else:
            return False
