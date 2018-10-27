# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 13:32:07 2018

@author: Joseph

This contains classes for reading and writing files in good format

"""

class FileOut():
    def __init__(self, filename, isBin):
        self.name=filename
        if isBin:
            write_type='wb'
        else:
            write_type='w'
        self.fout=open(filename+'.txt', write_type)
    
    # Header with information about file
    def header(self, title='Run'):
        self.fout.write('######################################################\n')
        self.fout.write('#              2D Navier-Stokes Solver               #\n')
        self.fout.write('#              Created by J. Mark Epps               #\n')
        self.fout.write('#          Part of Masters Thesis at UW 2018-2020    #\n')
        self.fout.write('######################################################\n\n')
        self.fout.write('############### '+title+' FILE #########################\n')
        self.fout.write('##########'+self.name+'##################\n\n')
    
    def input_writer(self, settings, BCs):
        self.fout.write('Settings:\n')
        keys=['Length','Width','Nodes_x','Nodes_y','Fluid','k','gamma','R','mu']
        for i in keys:
            self.fout.write(i)
            self.fout.write(':')
            self.fout.write(str(settings[i]))
            self.fout.write('\n')
        
        self.fout.write('\nMeshing details:\n')
        keys=['bias_type_x','bias_size_x','bias_type_y','bias_size_y']
        for i in keys:
            self.fout.write(i)
            self.fout.write(':')
            self.fout.write(str(settings[i]))
            self.fout.write('\n')
        
        self.fout.write('\nTime advancement:\n')
        keys=['CFL','total_time_steps', 'Time_Scheme']
        for i in keys:
            self.fout.write(i)
            self.fout.write(':')
            self.fout.write(str(settings[i]))
            self.fout.write('\n')
        
        self.fout.write('\nBoundary conditions:\n')
        keys=['bc_type_left', 'bc_left_rho', 'bc_left_u', 'bc_left_v', 'bc_left_p', 'bc_left_T',\
              'bc_type_right','bc_right_rho','bc_right_u','bc_right_v','bc_right_p','bc_right_T',\
              'bc_type_south','bc_south_rho','bc_south_u','bc_south_v','bc_south_p','bc_south_T',\
              'bc_type_north','bc_north_rho','bc_north_u','bc_north_v','bc_north_p','bc_north_T']
        for i in keys:
            self.fout.write(i)
            self.fout.write(':')
            self.fout.write(str(BCs[i]))
            self.fout.write('\n')
        self.fout.write('\n')
    
    def Write_timestep_data(self, timeStep, dt):
        self.fout.write('Time step: '+timeStep+'\n')
        self.fout.write('Time step size: '+dt+'\n\n')
        
    
    def Write_single_line(self, string):
        self.fout.write(string)
        self.fout.write('\n')
    
    def close(self):
        self.fout.close()
        
class FileIn():
    def __init__(self, filename, isBin):
        self.name=filename
        if isBin:
            read_type='rb'
        else:
            read_type='r'
        self.fout=open(filename+'.txt', read_type)