# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:52:09 2024

@author: smith
"""
import os
import json


class ResultWriter():
    def __init__(self,logfile,frequency=1):
        self.logfile=logfile
        self.frequency=frequency
        self.count=0
    def write(self,result,systparams=[]):
        if(self.count%self.frequency!=0):
            self.count+=1
            return
        if(not os.path.exists(self.logfile)):
            f=open(self.logfile,"w+")
            f.write("")
            f.close()
        f=open(self.logfile,"a")
        f.write(json.dumps(result)+"\n")
        f.close()


class mlogger():
        def __init__(self,fn):
                self.fn=fn
                if(not os.path.isfile(fn)):
                        f=open(self.fn,"w+")
                        f.close()
        def log(self,s):
                f=open(self.fn,"a")
                f.write(json.dumps(s)+"\n")
                f.close()