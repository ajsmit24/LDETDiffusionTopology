# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:34:39 2024

@author: smith
"""
import utils
mylogger=utils.mlogger("main.log")
log=mylogger.log

import random 



random.seed(0)
   
class RandomWalker():
    def __init__(self,graph,max_steps=100000,endnode="G",particle_initial_pos=None,
                 options={"periodic":True,"distance":"euclidean"}):
        self.graph=graph
        self.particle_location=particle_initial_pos
        self.max_steps=int(max_steps)
        self.endnode=endnode
        self.total_steps=0
        
        if("periodic" not in options):
            options["periodic"]=True
        if("distance" not in options):
            options["distance"]="euclidean"
            
        self.periodic=options["periodic"]
        self.distance_mode=options["distance"]
        self.options=options
        
        if(not self.periodic and 
           not ("suppress_warn" in self.options and "peroidic" in self.options["suppress_warn"])):
                raise Exception("ERROR:Implemented behavior is constistent for both graphs with "+
                                "edges that create a peroidic boundary and graphs that are truely "+
                                "finite. It is not clear what behavior you expect to differ by "+
                                "setting peroidic=False, but this error can be suppressed by "+
                                "passing options['suppress_warn']=['peroidic']. Please review the "+
                                "code in the random walker file before doing so. To get there "+
                                "follow the stack trace for this error")


        self.abs_displacement=0
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
    
        






if(__name__=="__main__"):
    pass
