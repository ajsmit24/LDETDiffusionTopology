# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:54:56 2024

@author: smith
"""
import numpy as np
import networkx as nx
import math
import random


#these lattice can have a number of properties of interest
##perhaps best is the mean first passage time to an edge from the center
##of a lattice with odd dimension lengths

class UniformNDLatticeConstructor():
    #supported options are center (generally for diffusion) or
    ##starting_surface for mean first passage
    def __init__(self,dims=[3,3,3],periodic=[True,True,True],particle_position="center"
                 ,options={}):
        self.periodic=periodic
        self.dimensions=dims
        self.graph=nx.grid_graph(dim=dims,periodic=periodic)
        self.particle_position=particle_position
        #self.insert_particle()
        self.options=options
                
    def draw(self):
        nx.draw(self.graph, with_labels = True)
        
    #starting_surface assumes the direction of flow will be that with the largest dimension
    #if all dimensions are equal size one will be chosen at random
    #or a dimension (spec with its index 0,1,2etc) can be set in options["flow"]
    def insert_particle(self):
        if(self.particle_position=="center"):
            self.particle_position=tuple(math.floor(d/2) for d in self.dimensions)
            self.graph.nodes[self.particle_position]["occ"]=1
        elif(self.particle_position=="starting_surface"):
            flow_dim=-1
            max_dim=max(self.dimensions)
            max_dims=[(i,self.dimensions[i]) for i in range(len(self.dimensions)) if(self.dimensions[i]==max_dim)]
            if(len(max_dims)==1):
                flow_dim=max_dims=[0][1]
            else:
                flow_dim=random.random.sample(max_dims,1)[0][1]
            if("flow" in self.options):
                flow_dim=self.options["flow"]
            if(flow_dim==-1):
                raise Exception("ERROR: flow value unexpectedly not set with particle_position="+
                                "starting_surface - in insert_particle of UniformNDLatticeConstructor ")  
        else:
            raise Exception("ERROR: Functionality insert_particle with particle_position="
                            +str(self.particle_position)+" -Has not yet been implemented in "+
                            "UniformNDLatticeConstructor")
        
        