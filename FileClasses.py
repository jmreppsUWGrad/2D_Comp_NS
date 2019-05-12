
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 13:32:07 2018

@author: Joseph

This contains classes for reading and writing files in good format

"""

import string as st

# Global declarations of data stored in files
keys_Settings=['Length','Width','Nodes_x','Nodes_y','Fluid','k','gamma','R','mu',\
               'Gravity_x','Gravity_y', 'bias_type_x', 'bias_size_x','bias_type_y','bias_size_y']
keys_Time_adv=['CFL','dt','total_time_steps', 'total_time', 'Time_Scheme','Number_Data_Output']

keys_BCs=     ['bc_type_left', 'bc_left_u', 'bc_left_v', 'bc_left_p', 'bc_left_T',\
              'bc_type_right','bc_right_u','bc_right_v','bc_right_p','bc_right_T',\
              'bc_type_south','bc_south_u','bc_south_v','bc_south_p','bc_south_T',\
              'bc_type_north','bc_north_u','bc_north_v','bc_north_p','bc_north_T']

class FileOut():
    def __init__(self, filename, isBin):
        self.name=filename
        if isBin:
            write_type='wb'
        else:
            write_type='w'
        self.fout=open(filename+'.txt', write_type)
    
    # Write a single string with \n at end
    def Write_single_line(self, string):
        self.fout.write(string)
        self.fout.write('\n')
    
    # Header with information about file
    def header(self, title='Run'):
        self.Write_single_line('######################################################')
        self.Write_single_line('#              2D Navier-Stokes Solver               #')
        self.Write_single_line('#              Created by J. Mark Epps               #')
        self.Write_single_line('#          Part of Masters Thesis at UW 2018-2020    #')
        self.Write_single_line('######################################################\n')
        self.Write_single_line('############### '+title+' FILE #########################')
        self.Write_single_line('##########'+self.name+'##################\n')
    
    def input_writer(self, settings, BCs):
        self.Write_single_line('Settings:')
        for i in keys_Settings:
            self.fout.write(i)
            self.fout.write(':')
            self.Write_single_line(str(settings[i]))
        
#        self.Write_single_line('\nMeshing details:')
#        keys=['bias_type_x','bias_size_x','bias_type_y','bias_size_y']
#        for i in keys:
#            self.fout.write(i)
#            self.fout.write(':')
#            self.Write_single_line(str(settings[i]))
##            self.fout.write('\n')
        
        self.Write_single_line('\nTime advancement:')
        for i in keys_Time_adv:
            self.fout.write(i)
            self.fout.write(':')
            self.Write_single_line(str(settings[i]))
        
        self.Write_single_line('\nBoundary conditions:')
        for i in keys_BCs:
            self.fout.write(i)
            self.fout.write(':')
            self.Write_single_line(str(BCs[i]))
        
#        self.fout.write('\nInitial conditions:\n')
#        self.Write_single_line('rho')
#        for i in range(len(rho[:,0])):
#            self.Write_single_line(str(rho[i,:]))
#        self.Write_single_line('rhou')
#        for i in range(len(rho[:,0])):
#            self.Write_single_line(str(rhou[i,:]))
#        self.Write_single_line('rhov')
#        for i in range(len(rho[:,0])):
#            self.Write_single_line(str(rhov[i,:]))
#        self.Write_single_line('rhoE')
#        for i in range(len(rho[:,0])):
#            self.Write_single_line(str(rhoE[i,:]))

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
        self.fin=open(filename+'.txt', read_type)
        
    def Read_Input(self, settings, BCs):
        for line in self.fin:
            if st.find(line, ':')>0 and st.find(line, '#')!=0:
                line=st.split(line, ':')
                if line[0] in keys_Settings:
                    if line[0]=='Fluid' or line[1]=='None\n':
                        # String handler
                        settings[line[0]]=st.split(line[1], '\n')[0]
                    elif line[0]=='Nodes_x' or line[0]=='Nodes_y':
                        # Integer values
                        settings[line[0]]=int(line[1])
                    else:
                        # Float values
                        settings[line[0]]=float(line[1])
                        
                elif line[0] in keys_Time_adv:
                    if line[0]=='Time_Scheme' or line[1]=='None\n':
                        # String handler
                        settings[line[0]]=st.split(line[1], '\n')[0]
                    elif line[0]=='total_time_steps' or line[0]=='Number_Data_Output':
                        settings[line[0]]=int(line[1])
#                    elif line[0]=='Output_directory':
#                        # String handler for directory
#                        settings[line[0]]=line[1]+':'+st.split(line[2], '\n')[0]
                    else:
                        # Float values
                        settings[line[0]]=float(line[1])
                        
                elif line[0] in keys_BCs:
                    if line[0]=='bc_type_left' or line[0]=='bc_type_right'\
                        or line[0]=='bc_type_south' or line[0]=='bc_type_north'\
                        or line[1]=='None\n':
                        # String handler
                        BCs[line[0]]=st.split(line[1], '\n')[0]
                    else:
                        try:
                            # Number for BC
                            BCs[line[0]]=float(line[1])
                        except:
                            # Grad tuple
                            BCs[line[0]]=(st.split(line[1],',')[0],float(st.split(line[1],',')[1]))
#                            BCs[line[0]]=st.split(line[1], '\n')[0]
                            
        self.fin.close()
        