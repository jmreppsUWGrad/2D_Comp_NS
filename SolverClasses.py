# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:17:11 2018

@author: Joseph

Solver classes for Compressible N-S equations. Takes in given object (geometry),
time step and convergence information and alters the object's temperature, 
velocity, pressure, density. BCs are applied as appropriate, but must be 
defined and copied into the solver object.

Features:
    -Conservative Fourrier number correction based on smallest discretization
    -


"""

import numpy
#import GeomClasses
#import MatClasses

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
    def __init__(self, geom_obj, CFL, timeSteps, conv):
        self.Domain=geom_obj # Geometry object
        self.CFL=CFL
        self.Nt=timeSteps
        self.conv=conv
        self.dx,self.dy=numpy.meshgrid(geom_obj.dx,geom_obj.dy)
#        self.dx=geom.dx        
#        self.dy=geom.dy
        self.maxCount=1000

        self.BCs={'BCx1': ('T',600,(0,-1)),\
                 'BCx2': ('T',300,(0,-1)),\
                 'BCy1': ('T',600,(0,-1)),\
                 'BCy2': ('T',300,(0,-1))\
                 }
        
        # Pointers to conservative/primitive variable arrays
        self.rhoE=geom_obj.rhoE # Conservative arrays
        self.rhou=geom_obj.rhou
        self.rhov=geom_obj.rhov
        self.rho=geom_obj.rho # Primitive arrays
        self.T=geom_obj.T
        self.u=geom_obj.u
        self.v=geom_obj.v
        self.p=geom_obj.p
    
    # Time step check with all dx and dys for stability (CHANGE TO CFL NUMBER CHECK)
    def getdt(self):
        dx=min(numpy.amin(self.dx),numpy.amin(self.dy))
        c=1 # ADD SPEED OF SOUND RETRIEVAL
        return self.CFL*dx/(c)
#        if self.Fo/dx**2<0.25:
#            print 'Time step is within stability limits'
#        else:
#            self.Fo=0.24999*dx**2
#            self.dt=self.Fo/self.Domain.mat_prop['k']*self.Domain.mat_prop['Cp']*self.Domain.mat_prop['rho']
#            print 'Time step changed to %3f for stability'%self.dt

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
        - source term
        - RK time advancement (eventually)
        
    """    
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