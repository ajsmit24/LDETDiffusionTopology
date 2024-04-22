# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:52:09 2024

@author: smith
"""
import os
import json


class ResultReader():
    def __init__(self,filename,suppress_warn=False):
        self.filename=filename
        self.suppress_warn=suppress_warn
    def read(self):
        if(not os.path.exists(self.filename)):
            return []
        f=open(self.filename,"r")
        txt=f.read()
        f.close()
        output=[]
        for line in txt.split("\n"):
            if(len(line)<3):
                continue
            try:
                output.append(json.loads(line))
            except Exception as e:
                if(not self.suppress_warn):
                    print("WARNING: could not json parse line:"+line+"-"+str(e))
        return output
                
                

class ResultWriter():
    def __init__(self,logfile,frequency=1,force_only=False,mute=False):
        self.logfile=logfile
        self.frequency=frequency
        self.count=0
        self.force_only=force_only
        self.mute=mute
    def write(self,result,systparams=[],force=False):
        if(self.mute):
            return
        if(self.count%self.frequency!=0):
            self.count+=1
            return
        if(self.force_only and (not force)):
            return 
        if(not os.path.exists(self.logfile)):
            f=open(self.logfile,"w+")
            f.write("")
            f.close()
        f=open(self.logfile,"a")
        f.write(json.dumps(result)+"\n")
        f.close()


def lattice_cast_node(node):
    if(type(node)==tuple):
        if(len(node)==1):
            return node[0]
        return node
    if(type(node)==int):
        return (node,)

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