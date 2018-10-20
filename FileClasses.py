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
    def header(self, title='Solver'):
        f=self.fout
        f.write('######################################################\n')
        f.write('#                      Solver                        #\n')
        f.write('#              Created by J. Mark Epps               #\n')
        f.write('#          Part of Masters Thesis at UW 2018-2020    #\n')
        f.write('######################################################\n')
    
    def Write(self, string):
        f=self.fout
        f.write(string)
        f.write('\n')