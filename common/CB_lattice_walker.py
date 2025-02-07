# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:24:00 2024

@author: smith
"""

#Standard python imports
import json
import random
import time
import argparse
import platform
import sys
import os

#custom import of my own code
import oop_cablebacteria_constructor
import utils
import convergence_criteria
##potentially depreciated custom code
import old_random_walker

mylogger=utils.mlogger("main.log")
log=mylogger.log

#note that this method is only preprocessing for CB relevant graphs
class GraphPreprocess():
    #j_inter_nodes is the number of nodes to insert between two junctions in the minimal topology
    ##model (the circuits based graphs) this defines the CG scale
    def __init__(self,nfiber,njunctions,random_particle_init=False,
                 j_inter_nodes=1,allow_warnings=True,CG_version="v1",length_radius_ratio=2):
        self.nfiber=nfiber
        self.njunctions=njunctions
        self.random_particle_init=random_particle_init
        self.j_inter_nodes=j_inter_nodes
        self.allow_warnings=allow_warnings
        self.length_radius_ratio=length_radius_ratio
        self.all_junction_related_nodes=[]
        self.fibers=None
        self.juncts=None
        
        self.recursion_limit_updated=False
        self.CG_version=CG_version
        if(self.nfiber==3):
            raise Exception("Can not conclusively determine junction nodes when nfiber=3")
            
    #__fiber_find_helper/find_fibers often even when well behaved exceed the recursion limit so
    ##we have to change the limit
    def update_recursion_limit(self):
        sys.setrecursionlimit(40000)
        self.recursion_limit_updated=True

    #note that here the start and stop are NOT the ends of the CB
    ##rather they are the edges of a single topology element 
    ##for example the nodes defining a singl einter-junction fiber segment
    #used mostly for adjusting scales of CG - this is actually fairly generic and future classes
    ##will likely copy from this
    def insert_n_nodes(self,graph,start_node,stop_node,n,mode="none"):
        #We need to remove the existing connection between the edges in order to create a new one
        #this helps us to avoid the following situation
        ##S---E should become S---X---X---X---E but would instead erroneously be
        ##-----------------
        ##|               |
        ##S---X---X---X---E Thus still having a single hop connection between S and E
        graph.remove_edge(start_node,stop_node)
        #get the maximum node number to help us keep number consistent
        last_node_num=max([n for n in graph.nodes if type(n)==int])
        #starting is actually an itterator changed in the loop so we use a new variable to 
        ##perserve information
        starting=start_node
        for i in range(n):
            #create a new node number
            new_node_number=last_node_num+1+i
            #link that new node to the previous one
            graph.add_edge(starting, new_node_number)
            if(i==n-1):
                #if this is the last one also connect it to the end
                graph.add_edge(new_node_number,stop_node)
            if(mode=="junct"):
                #if we are working on a CB junction then keep track of which junctions
                #the new nodes correspond to
                self.all_junction_related_nodes.append(new_node_number)
            #update what the "previous" node was
            starting=new_node_number

    def find_junction_nodes(self,graph):
        juncts=[]
        #if this has already been done dont do it again
        if(self.juncts!=None):
            return self.juncts
        for n in graph.nodes:
            #this degree condition defines what is to be a junction in most* cases
            if(graph.degree[n]==self.nfiber):
                juncts.append(n)
        #in some cases I want a more extensive list of non-fiber cells
        self.all_junction_related_nodes=juncts
        self.jucts=juncts
        return juncts
    
    #NOTE this is "__" method calling it directly may have unexpected affects
    ##It should generally be called from find_fibers
    #d is the current recursion depth which is usefull for debugging
    def __fiber_find_helper(self,nextnode,prev,graph,endnode="G",d=0):
        def isendnode(i):
            if(self.CG_version=="v2"):
                return "special" in graph.nodes[i] and graph.nodes[i]["special"]=="end"
            if(self.CG_version=="v1"):
                return i==endnode
        #this method is recursive and is can recure very deeply - often beyond the
        ## system limit - so we need to update it (but only at the top level call)
        if(d==0 and not self.recursion_limit_updated):
            self.update_recursion_limit()
        #this is the basecase as it is the end of the fiber and recursion used list concatination
        if(isendnode(nextnode)):
            return []
        else:
            for n in graph.neighbors(nextnode):
                #make sure we are iterating down the fiber and not traversing junctions
                #NOTE find_fibers alters all_junction_related_nodes so it does not contain
                ##the starting and ending nodes 
                if(n!=prev and (n not in self.all_junction_related_nodes)):
                    #recure and traverse further
                    return [n]+self.__fiber_find_helper(n,nextnode,graph,d=d+1)
        if(self.allow_warnings):
            print("WARNING: in __fiber_find_helper - MAL RETURN ")
        else:
            raise Exception("WARNING-ERROR: in __fiber_find_helper - MAL RETURN ")
    
    #highly reliant on calling __fiber_find_helper
    def find_fibers(self,graph,juncts):
        #do not recalculate if already stored
        if(self.fibers!=None):
           return self.fibers
        #find all junctions except start and end
        self.all_junction_related_nodes=[jn for jn in self.all_junction_related_nodes if(jn!="G" and jn!=1)]
        #start looking for fibers allong each of the spokes of the first junction
        fibers=[]
        if(self.CG_version=="v2"):
            fibers=[[n] for n in graph.nodes if ("special" in graph.nodes[n] and graph.nodes[n]["special"]=="start")]
        else:
            fibers=[[n] for n in graph.neighbors(1) if (n!='G' and n!=0)]
        for i in range(len(fibers)):
            fibers[i]+=self.__fiber_find_helper(fibers[i][0],1,graph)
        self.fibers=fibers
        return fibers
    #linearizes CB
    def delete_other_fibers(self,graph):
        juncts=self.find_junction_nodes(graph)
        fibers=self.find_fibers(graph,juncts)
        #removes all fiber nodes except the zeroth fiber and the start and end nodes also remain
        for i in range(len(fibers)):
           if(i!=0):
               for j in range(len(fibers[i])):
                   if(j!=0 and j!=len(fibers[i])-1):
                      graph.remove_node(fibers[i][j])
        self.fibers=[fibers[0]]
    #CB from the circuit generator have a circuit closing connector node that is general node 0
    #this is not appropriate for this hopping model so it is removed
    def remove_connector(self,graph,connector=0):
        graph.remove_node(connector)
    
    def add_between_nodes(self,graph):
        juncts=json.loads(json.dumps(self.find_junction_nodes(graph)))
        fibers=self.find_fibers(graph,juncts)
        for jn in juncts:
            jneighs=[n for n in graph.neighbors(jn)]
            for fn in jneighs:
                mode="junct"
                if(jn==1 or jn=="G"):
                    mode="none"
                self.insert_n_nodes(graph,jn,fn,self.j_inter_nodes,mode=mode)
        for fiber in fibers:
            for i in range(1,len(fiber)):
                #self.insert_n_nodes(graph,fiber[i-1],fiber[i],2*(self.j_inter_nodes+1)-1)
                self.insert_n_nodes(graph,fiber[i-1],fiber[i],self.length_radius_ratio*(self.j_inter_nodes+1)-1)
        
    def insert_particle(self,graph,particle_position=1):
        if(self.CG_version=="v2"):
            posible_nodes=[n for n in graph.nodes 
                           if("special" in graph.nodes[n] and
                              graph.nodes[n]["special"]=="start")]
            particle_position=random.sample(posible_nodes,1)[0]
            graph.nodes[particle_position]["occ"]=1
            return particle_position
        if(self.random_particle_init):
            posible_nodes=[n for n in graph.nodes]
            particle_position=random.sample(posible_nodes,1)[0]
            graph.nodes[particle_position]["occ"]=1
        else:
            graph.nodes[particle_position]["occ"]=1
        return particle_position
    
    def CG_v2_trim_ends(self,graph,startnode=1,endnode="G"):
        if(self.CG_version!="v2"):
            raise Exception("ERROR: mixing CG versions")
        for n in graph.neighbors(startnode):
            graph.nodes[n]["special"]="start"
        for n in graph.neighbors(endnode):
            graph.nodes[n]["special"]="end"
        graph.remove_node(startnode)
        graph.remove_node(endnode)
        
    
    def do_setup(self,graph,TrueOneDim=False):
        self.remove_connector(graph)
        if(self.CG_version=="v2"):
            self.CG_v2_trim_ends(graph)
        if(TrueOneDim):
            self.delete_other_fibers(graph)
        self.add_between_nodes(graph)
        self.insert_particle(graph)
            
   
class Handler():
    def __init__(self,nfiber,njuctions,numb_repeats,writer,max_steps=1e6,particle_initial_pos=1,savepath=False,j_inter_nodes=1,length_radius_ratio=2,CG_version="v2"):
        self.nfiber=nfiber
        self.njuctions=njuctions
        self.max_steps=max_steps
        self.particle_initial_pos=particle_initial_pos
        self.numb_repeats=numb_repeats
        self.lin_path_len=2*self.njuctions-1+4
        self.writer=writer
        self.j_inter_nodes=j_inter_nodes
        self.length_radius_ratio=length_radius_ratio
        self.CG_version=CG_version
    
    def do_one_round(self,bridged=True):
        cb=oop_cablebacteria_constructor.CableBacteria(self.nfiber,self.njuctions,bridged=bridged,CG_version=CG_version)
        cb.to_graph()
        gp=GraphPreprocess(self.nfiber,self.njuctions,j_inter_nodes=self.j_inter_nodes,length_radius_ratio=self.length_radius_ratio,CG_version=CG_version)
        gp.do_setup(cb.graph,TrueOneDim=not bridged)
        rw=old_random_walker.RandomWalker(cb.graph,max_steps=self.max_steps,particle_initial_pos=self.particle_initial_pos)
        rw.run_random_walk()
        return rw.total_steps
        
    def run_many(self):
        self.bridged_results=[]
        self.unbridged_results=[]
        for i in range(self.numb_repeats):
            self.bridged_results.append(self.do_one_round(bridged=True))
            self.unbridged_results.append(self.do_one_round(bridged=False))
        bridge_path_len=sum(self.bridged_results)/self.numb_repeats
        unbridge_path_len=sum(self.unbridged_results)/self.numb_repeats
        res={}
        res["BRIDGED"]=bridge_path_len
        res["UNBRIDGED"]=unbridge_path_len
        res["RATIO"]=bridge_path_len/unbridge_path_len
        self.writer.write(res)
        

#----------BEGIN-------------!!!IMPORTANT!!!----------BEGIN-------------
#the following section is highly bespoke solutions. They are integrated here for consistency
#this should be avoided in writing new code
#these are only here for backwards compatibility
#DEPRECIATED
#-----------------------------------------------------------------------

def MPI_runner(p):
    try:
        h,i,b=p
        log(json.dumps([i,b]))
        res={}
        rank=1
        if(__name__!="__main__"):
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        random.seed(rank*i)
        res["f"]=h.nfiber
        res["j"]=h.njuctions
        res["b"]=b
        res["i"]=i
        start=time.time()
        res["out"]=h.do_one_round(bridged=b)
        res["time"]=time.time()-start
        h.writer.write(res)
        return None
    except Exception as e:
        log(str(e))
        return str(e)

def CB_argparser():
    parser = argparse.ArgumentParser(description="Script with command-line arguments")
    # Add arguments for nfibers, njuncts, nperround, tol, and sub-script
    parser.add_argument("--nfibers", type=int, default=50, help="Number of fibers (integer)")
    parser.add_argument("--njuncts", type=int,  default=10, help="Number of junctions (integer)")
    parser.add_argument("--nperround", type=int, default=10, help="Number per round (integer)")
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance (float)")
    parser.add_argument("--sub-script", type=str, default="out.out", help="Sub-script (string)")
    parser.add_argument("--cof-scale", type=int, default=1, help="scaling for cofactors based on number connecting to junction")
    args = parser.parse_args()

    nfibers = args.nfibers
    njuncts = args.njuncts
    nperround = args.nperround
    tol = args.tol
    sub_script = args.sub_script
    cof_scale=args.cof_scale
    f=nfibers
    j=njuncts
    n=nperround
    tol=tol

    conv_step=n
    computed=0
    is_conv=False
    past_res=[]
    init_size=n
    fn=sub_script.split(".")[0]+".out"
    rw=utils.ResultWriter(fn)
    h=Handler(f,j,n,rw,max_steps=1e12,j_inter_nodes=cof_scale)
    log(fn)
    infologger=utils.mlogger(sub_script.split(".")[0]+".info")
    start=time.time()
    infologger.log("Start:"+str(start))
    while(not is_conv):
        params=[[h,i,True] for i in range(computed,computed+conv_step)]+[[h,i,False] for i in range(computed,computed+conv_step)]
        result=[]
        if(platform.system()!="Windows"):
            from mpi4py.futures import MPIPoolExecutor
            with MPIPoolExecutor() as pool:
                result = pool.map(MPI_runner, params)
        else:
            result=[MPI_runner(p) for p in params]
        computed+=conv_step
        log(computed-conv_step)
        is_conv,new_past_res=convergence_criteria.old_ratio_conv(past_res,fn,conv_thresh=tol)
    log("CONV AT:"+str(computed))
    endt=time.time()
    infologger.log("END:"+str(endt))
    infologger.log("ELAPSED:"+str(endt-start))
    infologger.log("THE_JOB_IS_DONE")
    #sys.excepthook = sys.__excepthook__
    
    
#----------END-------------!!!IMPORTANT!!!----------END-------------

         
