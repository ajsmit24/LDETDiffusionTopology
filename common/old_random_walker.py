

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:16:36 2023

@author: smith
"""
import utils
mylogger=utils.mlogger("main.log")
log=mylogger.log

import random 
import networkx as nx
import json
import os
import sys
import time



random.seed(0)
   
class RandomWalker():
    def __init__(self,graph,max_steps=100000,endnode="G",particle_initial_pos=None):
        self.graph=graph
        self.particle_location=particle_initial_pos
        self.max_steps=int(max_steps)
        self.endnode=endnode
        self.total_steps=0
        if(particle_initial_pos==None):
            self.particle_location=self.find_initial_particle
        
    def find_initial_particle(self):
        pass
    
    def step(self):
        pos_steps=[n for n in self.graph.neighbors(self.particle_location)]
        new_position=random.sample(pos_steps,1)[0]
        self.graph.nodes[self.particle_location]["occ"]=0
        self.graph.nodes[new_position]["occ"]=1
        self.particle_location=new_position
        self.total_steps+=1
    
    def run_random_walk(self):
        for i in range(self.max_steps):
            self.step()
            if(self.particle_location==self.endnode):
                return
        print("PASSED MAX STEPS")
    
    def print_outcome(self):
        print(self.total_steps)

def MPI_runner(params):
    pass
    
        

def check_conv(past,fn,conv_thresh=0.001):
    thresh=conv_thresh
    log("checking conv")
    f=open(fn,"r")
    txt=f.read()
    f.close()
    data=[]
    datadict={}
    has_number={}
    for line in txt.split("\n"):
        if(len(line)>3):
            data.append(json.loads(line))
    for item in data:
        if(not item["i"] in datadict):
            datadict[item["i"]]=0
        if(not item["i"] in has_number):
            has_number[item["i"]]=0
        has_number[item["i"]]+=1
        if(item["b"]):
            if(datadict[item["i"]]==0):
                datadict[item["i"]]=item["out"]
            else:
                datadict[item["i"]]*=item["out"]
        else:
            if(datadict[item["i"]]==0):
                datadict[item["i"]]=1/item["out"]
            else:
                datadict[item["i"]]*=1/item["out"]
    keylist=[i for i in has_number if(has_number[i]==2)]
    keylist=sorted(keylist)
    for i in keylist:
        if(datadict[i]>100):
            datapair=[]
            for item in data:
                if(item["i"]==i):
                    datapair.append(item["i"])
            log("LOGGING STRANGNESS")
            log("Strange Ratio: "+str(datadict[i])+" "+str(datapair)+"PARAMS:"+json.dumps(list(sys.argv)))
            #raise Exception("Strange Ratio: "+str(datadict[i])+" "+str(datapair))
    datalist=[datadict[i] for i in keylist]
    tlist=[0]
    for i in range(1,len(datalist)):
        tlist.append((tlist[i-1]*(i)+datalist[i])/(i+1))
    res=[]
    for i in range(1,len(tlist)):
        res.append(abs(tlist[i]-tlist[i-1]))
    return (sum(res[-10:])/10)<thresh,"dep"




if(__name__=="__main__"):
    pass
