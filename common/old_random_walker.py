

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:16:36 2023

@author: smith
"""
import utils
mylogger=utils.mlogger("main.log")
log=mylogger.log

import random 



random.seed(0)
   
class RandomWalker():
    def __init__(self,graph,max_steps=100000,endnode="G",particle_initial_pos=None,CG_version="v2"):
        self.graph=graph
        self.particle_location=particle_initial_pos
        self.max_steps=int(max_steps)
        self.endnode=endnode
        self.total_steps=0
        self.CG_version=CG_version
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
            if(self.CG_version=="v2"):
                if("special" in self.graph.nodes[self.particle_location] and
                   self.graph.nodes[self.particle_location]["special"]=="end"):
                    return 
            else:
                if(self.particle_location==self.endnode):
                    return
        print("PASSED MAX STEPS")
    
    def print_outcome(self):
        print(self.total_steps)

def MPI_runner(params):
    pass
    
        






if(__name__=="__main__"):
    pass
