# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 13:32:07 2018

@author: Joseph

This contains classes for outputting files in good format

"""

class FileOut():
    def __init__(self, filename, isBin):
        self.name=filename
        if isBin:
            write_type='wb'
        else:
            write_type='w'
        self.fout=open(filename, write_type)
    def header(self, title='Run'):
        self.fout.write('######################################################\n')
        self.fout.write('#              2D Navier-Stokes Solver               #\n')
        self.fout.write('#              Created by J. Mark Epps               #\n')
        self.fout.write('#          Part of Masters Thesis at UW 2018-2020    #\n')
        self.fout.write('######################################################\n')
        self.fout.write(title)
    
    def Write(self, string):
        self.fout.write(string)
        self.fout.write('\n')