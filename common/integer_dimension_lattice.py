# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:54:56 2024

@author: smith
"""
import numpy as np
import networkx as nx
import math
import random
import itertools

import utils


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
        self.avail_positions=["center"]
        self.reorientation={}
        if(particle_position in self.avail_positions):
            self.insert_particle()
        self.options=options
        
        
                
    def draw(self):
        nx.draw(self.graph, with_labels = True)
        
    #fun fact networkx reorders the dimensions if they arent uniform
    #ie if I make a [3,5] lattice the point (2,4) does not exist but (4,2) does
    #so we need to store instead that the dimensions of that lattice were ordered
    #ie 5,3 instead of 3,5     
    def orient_dimensions(self):
        nodelist=self.graph.nodes
        if(len(self.dimensions)>5):
            print("WARNING: method orient_dimensions scales very poorly with number of "+
                  "dimensions. please consider passing the dimensions in the correct order"+
                  " and editing this code")
        all_perms=itertools.permutations(zip(self.dimensions,list(range(len(self.dimensions)))),len(self.dimensions))
        pos_dim_sizes=[]
        added_perms=set()
        for perm in all_perms:
            temp=tuple(p[0] for p in perm)
            if(temp not in  added_perms):
                added_perms.add(temp)
                pos_dim_sizes.append([temp,tuple(p[1] for p in perm)])
        reorientation_list=[]
        for pos_dim_info in pos_dim_sizes:
            pd=pos_dim_info[0]
            temp=[]
            if(len(self.dimensions)<2):
                temp=utils.lattice_cast_node(tuple(pdi-1 for pdi in pd),target=int)
            else:
                temp=tuple(pdi-1 for pdi in pd)
            if(temp in nodelist):
                self.dimensions=list(pd)
                reorientation_list=pos_dim_info[1]
                break
        for i in range(len(reorientation_list)):
            self.reorientation[reorientation_list[i]]=i
                
        new_peroidic=[]
        for i in range(len(self.periodic)):
            new_peroidic.append(self.periodic[self.reorientation[i]])
        self.periodic=new_peroidic
        return
        
    #starting_surface assumes the direction of flow will be that with the largest dimension
    #if all dimensions are equal size one will be chosen at random
    #or a dimension (spec with its index 0,1,2etc) can be set in options["flow"]
    def insert_particle(self):
        if(self.particle_position=="center"):
            self.orient_dimensions()
            self.particle_position=utils.lattice_cast_node(tuple(math.floor(d/2) for d in self.dimensions))
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
        