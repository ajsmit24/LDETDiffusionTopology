# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:34:39 2024

@author: smith
"""
import utils
import inspect
import math
mylogger=utils.mlogger("main.log")
log=mylogger.log

import random 



random.seed(0)



   
class RandomWalker():
    #note endnode and maxsteps should not be used and are only here for
    ##backwards compatibility
    def __init__(self,graph,particle_initial_pos=None,endcriteria={"single_node_boundary":"G"},
                 options={"periodic":True,"distance":"euclidean"},endnode="G",max_steps=1e94):
       
        #validation of input parameters
        self.valid_options={"options[graphType]":
                            ["defualt",#any graph other than those listed below
                             "lattice",#any graph with tuple node labels but designed for
                             ##those specifed in integer_dimension_lattice
                             #"options[latticeObj]" required for lattice graphs generally a 
                             ##integer_dimension_lattice object
                             ##please see ~~latticeObj option handling~~ for greater detail
                             "dynamic_deterimination"#default value of the parameter - attempts to determine
                             #graph type based on the data type of node labels
                                ],
                            "options[periodic]":[True], #based on the idea that somethings may change
                            #-particularly distance measurement-change on a peroidic graph
                            #this is currently not used see ~~peroidic option handling~~
                            "options[distance]":["euclidean"],#
                            "endcriteria":["single_node_boundary","surface_boundary","conv_method","conv_class"],
                            "endcriteria[surface_boundary]":["bound","writer"]
                            }
        
        #setup simple parameters
        self.graph=graph
        self.particle_location=particle_initial_pos
        self.max_steps=int(max_steps)
        self.endnode=endnode
        self.total_steps=0
        
        
        #setup default values of options
        #this is used for the case that the user passes some options
        ##that do not include every option
        if("periodic" not in options):
            options["periodic"]=True
        if("distance" not in options):
            options["distance"]="euclidean"
        if("suppress_err" not in options):
            options["suppress_err"]=[]
        if("graphType" not in options):
            options["graphType"]="dynamic_deterimination"
            
        self.periodic=options["periodic"]
        self.distance_mode=options["distance"]
        self.graphType=options["graphType"]
        self.options=options
        
        #validate restricted options
        criteria_count=sum([1 for c in self.valid_options["endcriteria"] if c in endcriteria])
        if(criteria_count!=1):
            raise Exception("ERROR:Passed either too many or too few endcriteria")
        if("conv_method" in endcriteria):
            if(str(type(endcriteria['conv_method']))!="<class 'function'>"):
                raise Exception("ERROR: endcriteria[conv_method] must be a function")
        if("surface_boundary" in endcriteria):
            k_req={"bound":"a float that is the distance to the surface","writer":"a util resultwriter class"}
            for k in self.valid_options["endcriteria[surface_boundary]"]:
                if(k not in endcriteria["surface_boundary"]):
                    raise Exception("ERROR: endcriteria[surface_boundary] must be a dictionary that contains the key"+
                                    k+"| endcriteria[surface_boundary]["+k+"] must be "+k_req[k])
        if("conv_class" in endcriteria):
            if("check_conv" not in dir(endcriteria["conv_class"])):
                raise Exception("ERROR: endcriteria[conv_class] must be an object instance of a class with "+
                                "the member method .check_conv")
        self.endcriteria=endcriteria
        ##based on the values in self.valid_options
        #~~peroidic option handling~~
        if((not self.valid_options["options[periodic]"] ) and 
           not ("peroidic" in self.options["suppress_err"])):
                raise Exception("ERROR:Implemented behavior is constistent for both graphs with "+
                                "edges that create a peroidic boundary and graphs that are truly "+
                                "finite. It is not clear what behavior you expect to differ by "+
                                "setting peroidic=False, but this error can be suppressed by "+
                                "passing options['suppress_err']=['peroidic']. Please review the "+
                                "code in the random walker file before doing so. To get there "+
                                "follow the stack trace for this error")
        if(self.distance_mode not in self.valid_options["options[distance]"]):
            raise Exception("ERROR: other distances metrics have have not been implemented")
            
        #determin graph type dynamically
        if(self.graphType=="dynamic_deterimination"):
            #not clear why the intermediate variable is needed but I swear it is needed
            self.graphTyp=RandomWalker.determine_graph_type(self.graph)
            x=RandomWalker.determine_graph_type(self.graph)
            self.graphType=x
            print("determined graph type:"+self.graphType)
        
        #check that the necissary options where passed to handle a given graph type
        #~~latticeObj option handling~~
        if("latticeObj" not in self.options):
            raise Exception("ERROR: options['latticeObj'] required for graph type lattice. Please "+
                            "note that latticeObj must have a .dimensions member that is a list "+
                            "describing the length of all dimensions. Most importantly this includes "+
                            "the finite length representing periodic dimensions as this is required "+
                            "to compute pbc corrected positions.")

        #setup handling of pbc
        self.pbc_passes=[]
        if(particle_initial_pos==None):
            self.particle_location=self.find_initial_particle()
        self.starting_position=self.particle_location
        self.particle_pos_pbc_corrected=self.particle_location
       
    def determine_graph_type(graph):
        n=list(graph.nodes)[0]
        if(type(n)==tuple):
            return "lattice"
        else:
            return "default"
        
        
    def find_initial_particle(self):
        raise Exception("ERROR: method find_initial_particle has not been implemented")
    
    def compute_distance(self,x,y):
        if (self.distance_mode=="euclidean"):
            if(self.graphType!="lattice" and ('noneuclid_dist' not in self.options['suppress_err'])):
                raise Exception("ERROR: distance measurment for non-lattice graphs only defined interms of "+
                                "node numbers and is not necessarily actual distances. NOTE: this error "+
                                "can be suppressed by options['suppress_err']=['noneuclid_dist']")
            else:
                if(type(x)==int):
                    x=[x]
                    y=[y]
                dist_sq=0
                for i in range(len(x)):
                    dist_sq+=(x[i]-y[i])**2
                return math.sqrt(dist_sq)
                
        else:
            raise Exception("ERROR: method distance_mode!='euclidean' not implemented in compute_distance")            
            
    #NOTE this method is only currently implemented rigously for lattice graphs
    ##this requires options['latticeObj'].dimensions please see ~latticeObj option handling~~
    ##in the class constructor
    def __step_account_for_pbc(self,old_point,new_point):
        if(self.graphType=="lattice"):
            old_point=utils.lattice_cast_node(old_point)
            new_point=utils.lattice_cast_node(new_point)
            #setup the pbc crossing tracker if it hasnt been setup yet
            if(self.pbc_passes==[]):
                self.pbc_passes=[0]*len(old_point)
            new_pos_pbc_cor=new_point
            for i in range(len(old_point)):
                #NOTE the order of this subtraction is important
                ##it is key for deteriming the sign of the change in pbc_passes
                #NOTE away to determine if an edge is a pbc edge all dimensions will be the same
                ##except one dimension will differ and that difference will be greater than 1
                ##this condition works for simple uniform integer dimension lattices
                pos_dif=old_point[i]-new_point[i]
                if(abs(pos_dif)>1):
                    #track which direction we crossed the pbc
                    ##for sake of example consider
                    ##l=UniformNDLatticeConstructor(dims=[5,2],periodic=[True,False])
                    ##l.draw()
                    ###we start on node (0,0)
                    ###stepping across the pbc to (0,4) our pbc corrected position
                    ###should be (0,-1) and pos_dif=-4 this gives us a -1 pbc cross in the
                    ###the second dimension as such we do -1* 2nd dimension length (5)
                    ###we get a pbc corrected position in the second dim of 4-5 =-1
                    ###for a total corrected position of (0,-1) 
                    ###---
                    ###Alternately if we start at (1,4) and cross the pbc to (1,0)
                    ###the pbc corrected position should be (1,5) we see that 
                    ###pos_dif=4 this gives us a +1 pbc cross in 2nd dim
                    ###as such our pbc correct 2nd dim position is 0+5 (5 is second dim length)
                    ###thus our pbc corrected position is (1,5)
                    if(pos_dif<0):
                        self.pbc_passes[i]-=1
                    else:
                        self.pbc_passes[i]+=1
            #calculate the pbc corrected possition
            new_pos_pbc_cor=tuple(new_pos_pbc_cor[i]+self.pbc_passes[i]*self.options["latticeObj"].dimensions[i] for i in range(len(new_pos_pbc_cor)))
            return utils.lattice_cast_node(new_pos_pbc_cor)
        else:
            #handle non-lattice graph pbc
            if("pbc_account_step" not in self.options["suppress_err"]):
                raise Exception("ERROR: PBC accounting for non-lattice graphs is not currently supported. "+
                                "Please consider modifying your local version of this file to include the "+
                                "necissary functionality. This error can also be suppressed by "+
                                "passing options['suppress_err']=['pbc_account_step']. Note that suppression "+
                                "results in this method simply returning the new position with no pbc correction. "+
                                "Follow the stack trace to this position in the source and the constructor for "+
                                "more details.")
            return new_point
            
    
    def step(self):
        #get possible next position
        pos_steps=[n for n in self.graph.neighbors(self.particle_location)]
        #select the next position to move to
        new_position=random.sample(pos_steps,1)[0]
        #update the graph occupancy and position data
        self.graph.nodes[self.particle_location]["occ"]=0
        self.graph.nodes[new_position]["occ"]=1
        #compute a new pbc adjusted position
        self.particle_pos_pbc_corrected=self.__step_account_for_pbc( self.particle_location,new_position)
        self.particle_location=new_position
        self.distance=self.compute_distance(self.starting_position, self.particle_pos_pbc_corrected)
        #increment step count
        #even if max steps isnt the stopping criteria this will still be used
        ##it does not have an impact generally as max steps is extremely high
        ##still if max steps is unexpectedly reached an error is thrown
        self.total_steps+=1
    
    def run_conv_checker(self):
        should_exit=False
        if("single_node_boundary" in self.endcriteria):
            if(self.particle_location==self.endnode):
                should_exit=True
                return should_exit
        if("surface_boundary" in self.endcriteria):
            if(self.distance>=self.endcriteria["surface_boundary"]["bound"]):
                should_exit=True
                self.endcriteria["surface_boundary"]["writer"].write({"mft":self.total_steps,
                                                                      "pbc_pos":self.particle_pos_pbc_corrected,
                                                                      "dist":self.distance,
                                                                      "isconv":True
                                                                      })
                return should_exit
        if("conv_class" in self.endcriteria):
            should_exit=self.endcriteria["conv_class"].check_conv(self)
            return should_exit

    def run_random_walk(self):
        for i in range(self.max_steps):
            self.step()
            if(self.run_conv_checker()):
                return
            
            
        if("max_step" not in self.options["suppress_err"]):
            raise Exception("ERROR:exceeded max steps in random walk. note that if this "+
                            "is expected this error can be supressed to a warning by "+
                            "passing options['suppress_err']=['max_step']. Follow "+
                            "stack trace to source and class constructor for more details.")
        else:
            print("WARN:exceeded max steps in random walk")
    
    def print_outcome(self):
        print(self.total_steps)

    





def MPI_runner(params):
    raise Exception("ERROR: not implemented. You may be looking for CB_lattice_walker.MPI_runner")
    
        






if(__name__=="__main__"):
    raise Exception("ERROR: not implemented. TODO link argparsers")
