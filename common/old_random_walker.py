

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:16:36 2023

@author: smith
"""
import logger
import platform
mylogger=logger.mlogger("main.log")
log=mylogger.log

import random 
import networkx as nx
import oop_cablebacteria_constructor
import json
import argparse
import os
import sys
import argparse
import time

sys.setrecursionlimit(40000)


random.seed(0)

class GraphPreprocess():
    def __init__(self,nfiber,njunctions,random_particle_init=False,j_inter_nodes=1):
        self.nfiber=nfiber
        self.njunctions=njunctions
        self.random_particle_init=random_particle_init
        self.j_inter_nodes=j_inter_nodes
        
        self.all_junction_related_nodes=[]
        self.fibers=None
        self.juncts=None
        if(self.nfiber==3):
            raise Exception("Can not conclusively determine junction nodes when nfiber=3")
        
    def insert_n_nodes(self,graph,start_node,stop_node,n,mode="none"):
        graph.remove_edge(start_node,stop_node)
        last_node_num=max([n for n in graph.nodes if type(n)==int])
        starting=start_node
        for i in range(n):
            new_node_number=last_node_num+1+i
            graph.add_edge(starting, new_node_number)
            if(i==n-1):
                graph.add_edge(new_node_number,stop_node)
            if(mode=="junct"):
                self.all_junction_related_nodes.append(new_node_number)
                
            starting=new_node_number

    def find_junction_nodes(self,graph):
        juncts=[]
        if(self.juncts!=None):
            return self.juncts
        for n in graph.nodes:
            if(graph.degree[n]==self.nfiber):
                juncts.append(n)
        self.all_junction_related_nodes=juncts
        self.jucts=juncts
        return juncts
    
    def __fiber_find_helper(self,nextnode,prev,graph,endnode="G",d=0):
        if(nextnode==endnode):
            return []
        else:
            for n in graph.neighbors(nextnode):
                if(n!=prev and (n not in self.all_junction_related_nodes)):
                    return [n]+self.__fiber_find_helper(n,nextnode,graph,d=d+1)
        print("MAL RETURN ")
    def find_fibers(self,graph,juncts):
        if(self.fibers!=None):
           return self.fibers
        self.all_junction_related_nodes=[jn for jn in self.all_junction_related_nodes if(jn!="G" and jn!=1)]
        fibers=[[n] for n in graph.neighbors(1) if (n!='G' and n!=0)]
        for i in range(len(fibers)):
            fibers[i]+=self.__fiber_find_helper(fibers[i][0],1,graph)
        self.fibers=fibers
        return fibers
    def delete_other_fibers(self,graph):
        juncts=self.find_junction_nodes(graph)
        fibers=self.find_fibers(graph,juncts)
        for i in range(len(fibers)):
           if(i!=0):
               for j in range(len(fibers[i])):
                   if(j!=0 and j!=len(fibers[i])-1):
                      graph.remove_node(fibers[i][j])
        self.fibers=[fibers[0]]
    def remove_connector(self,graph,connector=0):
        graph.remove_node(connector)
    
    def add_between_nodes(self,graph):
        juncts=json.loads(json.dumps(self.find_junction_nodes(graph)))
        
        for jn in juncts:
            jneighs=[n for n in graph.neighbors(jn)]
            for fn in jneighs:
                mode="junct"
                if(jn==1 or jn=="G"):
                    mode="none"
                self.insert_n_nodes(graph,jn,fn,self.j_inter_nodes,mode=mode)
        fibers=self.find_fibers(graph,juncts)
        for fiber in fibers:
            for i in range(1,len(fiber)):
                self.insert_n_nodes(graph,fiber[i-1],fiber[i],2*(self.j_inter_nodes+1)-1)
        
    def insert_particle(self,graph,particle_position=1):
        if(self.random_particle_init):
            posible_nodes=[n for n in graph.nodes]
            particle_position=random.sample(posible_nodes,1)[0]
            graph.nodes[particle_position]["occ"]=1
        else:
            graph.nodes[particle_position]["occ"]=1
        return particle_position
    
    def do_setup(self,graph,TrueOneDim=True):
        self.remove_connector(graph)
        if(TrueOneDim):
            self.delete_other_fibers(graph)
        self.add_between_nodes(graph)
        self.insert_particle(graph)
            
        
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
   
class Handler():
    def __init__(self,nfiber,njuctions,numb_repeats,writer,max_steps=1e6,particle_initial_pos=1,savepath=False,j_inter_nodes=1):
        self.nfiber=nfiber
        self.njuctions=njuctions
        self.max_steps=max_steps
        self.particle_initial_pos=particle_initial_pos
        self.numb_repeats=numb_repeats
        self.lin_path_len=2*self.njuctions-1+4
        self.writer=writer
        self.j_inter_nodes=j_inter_nodes
    
    def do_one_round(self,bridged=True):
        #print(self.nfiber,self.njuctions)
        cb=oop_cablebacteria_constructor.CableBacteria(self.nfiber,self.njuctions,bridged=bridged)
        cb.to_graph()
        gp=GraphPreprocess(self.nfiber,self.njuctions,j_inter_nodes=self.j_inter_nodes)
        gp.do_setup(cb.graph)
        rw=RandomWalker(cb.graph,max_steps=self.max_steps,particle_initial_pos=self.particle_initial_pos)
        rw.run_random_walk()
        #print("-"*10+"Path length"+"-"*10)
        #rw.print_outcome()
        #print("linear length:",self.lin_path_len)
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
    
        
class ResultWriter():
    def __init__(self,logfile):
        self.logfile=logfile
    def write(self,result,systparams=[]):
        if(not os.path.exists(self.logfile)):
            f=open(self.logfile,"w+")
            f.write("")
            f.close()
        f=open(self.logfile,"a")
        f.write(json.dumps(result)+"\n")
        f.close()

        

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


def main():
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
    rw=ResultWriter(fn)
    h=Handler(f,j,n,rw,max_steps=1e12,j_inter_nodes=cof_scale)
    log(fn)
    infologger=logger.mlogger(sub_script.split(".")[0]+".info")
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
        is_conv,new_past_res=check_conv(past_res,fn,conv_thresh=tol)
    log("CONV AT:"+str(computed))
    endt=time.time()
    infologger.log("END:"+str(endt))
    infologger.log("ELAPSED:"+str(endt-start))
    infologger.log("THE_JOB_IS_DONE")
    #sys.excepthook = sys.__excepthook__

if(__name__=="__main__"):
    main()
