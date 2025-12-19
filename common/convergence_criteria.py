# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:50:41 2024

@author: smith
"""

import json
import sys
import math
import numpy as np

from welford import Welford
#https://github.com/a-mitani/welford

import utils


global_default_thresh=0.05

#TODO COMMENT THIS FILE


class Rolling_Av_Conv():
    def mean_str(k):
        return "<"+k+">"
    def __init__(self,property_key,threshold_type="relative_std",threshold_limit=global_default_thresh,usefile=True):
        self.threshold_types=["relative_std","absolute_value"]
        if(threshold_type not in self.threshold_types):
            raise Exception("ERROR: Invalid argument value threshold_type="+threshold_type)
            
        self.threshold_type=threshold_type
        self.threshold_limit=threshold_limit
        self.property_key=property_key
        self.usefile=usefile
       
        if(self.threshold_type!="relative_std"):
            raise Exception("ERROR:threshold_types!=relative_std has not been implemented. Specified value:"+self.threshold_type)
        self.running_stats={}
        self.mean_props=[Rolling_Av_Conv.mean_str(k) for k in self.property_key]
        for k in self.property_key:
            self.running_stats[k]=Welford()
            self.running_stats[Rolling_Av_Conv.mean_str(k)]=Welford()
         
            
    def check_conv(self,data,verbose=True):
        for i in range(len(data)):
            if(self.usefile):
                reader=utils.ResultReader(data[i])
                for line in reader.read():
                    if(line["isconv"]):
                        for k in self.property_key:
                            self.running_stats[k].add(np.array(line[k]))
                            means=self.running_stats[k].mean.item()
                            self.running_stats[Rolling_Av_Conv.mean_str(k)].add(np.array(means))
            else:
                for k in self.property_key:
                    self.running_stats[k].add(np.array(data[i][k]))
                    means=self.running_stats[k].mean.item()
                    self.running_stats[Rolling_Av_Conv.mean_str(k)].add(np.array(means))
                        
            means={}
            abs_std={}
            rel_stds={}
            is_all_conv=True
            for k in self.mean_props:
            #for k in self.property_keys:
              means[k]=self.running_stats[k].mean.item()
              abs_std[k]=math.sqrt(self.running_stats[k].var_s.item())
              #abs_std=self.running_stats[Rolling_Av_Conv.mean_str(k)]
              rel_stds[k]=abs_std[k]/means[k]
              is_all_conv=is_all_conv and rel_stds[k]<self.threshold_limit
            if(not verbose):
                return is_all_conv
            else:
                #return is_all_conv,rel_stds
                return is_all_conv,{"rel_stds":rel_stds,"means":means,"abs_std":abs_std}
                    




#a more general version of Diffusion_Conv
class Generalized_Conv():
    def __init__(self,property_keys,threshold_type="relative_std",threshold_limit=global_default_thresh):
        self.threshold_types=["relative_std","absolute_value"]
        if(threshold_type not in self.threshold_types):
            raise Exception("ERROR: Invalid argument value threshold_type="+threshold_type)
        self.threshold_type=threshold_type
        self.threshold_limit=threshold_limit
        self.threshold=threshold_limit
        self.property_keys=property_keys
        self.running_stats={}
        for property_key in property_keys:
           self.running_stats[property_key]=Welford()
    
    
    def check_conv(self,output_files,verbose=True):
        if(self.threshold_type!="relative_std"):
            raise Exception("ERROR:threshold_types!=relative_std has not been implemented. Specified value:"+self.threshold_type)
        for of in output_files:
            reader=utils.ResultReader(of)

            for line in reader.read():
                if(line["isconv"]):
                    for k in self.running_stats:
                        self.running_stats[k].add(np.array(line[k]))
            
            is_all_conv=True         
            rel_stds={}
            means={}
            abs_std={}
            for k in self.running_stats:
                means[k]=self.running_stats[k].mean.item()
                abs_std[k]=self.running_stats[k].var_s.item()
                rel_stds[k]=abs_std[k]/means[k]
                is_all_conv=is_all_conv and rel_stds[k]<self.threshold
                
            if(not verbose):
                return is_all_conv
            else:
                #return is_all_conv,rel_stds
                return is_all_conv,{"rel_stds":rel_stds,"means":means,"abs_std":abs_std}

class Diffusion_Conv():
    def __init__(self,run_id,result_writer,graph_dim,threshold_type="relative_std",
                 threshold_limit=global_default_thresh,abs_min_steps=25):
        self.threshold_types=["relative_std","absolute_value"]
        if(threshold_type not in self.threshold_types):
            raise Exception("ERROR: Invalid argument value threshold_type="+threshold_type)
        self.threshold_type=threshold_type
        self.threshold_limit=threshold_limit
        self.graph_dim=graph_dim
        self.result_writer=result_writer
        self.run_id=run_id
        self.abs_min_steps=abs_min_steps
        self.last_dif=None
        self.info=[]
        
        self.conv_check_count=0
        
        self.running_stats={"<x^2>":Welford(),"D":Welford()}
        #for i in range(self.graph_dim):
        #    self.running_stats[str(i)]=Welford()
        
    def calc_mean_and_var(self,new_data,stat_type):
        for nd in new_data:
            self.running_stats[stat_type].add(np.array(nd))
        return self.running_stats[stat_type].mean.item(), self.running_stats[stat_type].var_s.item()
    def update_diffusion_coef(self,time,distance):
        #msd=mean squared displacement
        msd_x2,std_x2=self.calc_mean_and_var([distance],"<x^2>")
        D=msd_x2/((2**self.graph_dim)*time)
        self.last_dif=D
        mean_D,std_D=self.calc_mean_and_var([D],"D")
        return {"D_cur":D,"<x^2>":{"mean":msd_x2,"std":std_x2},"D":{"mean":mean_D,"std":std_D}}
    def get_diffusion(self):
        return self.last_dif
    def check_conv(self,rand_walk):
        stats=self.update_diffusion_coef(rand_walk.total_steps,rand_walk.distance**2)
        if(self.threshold_type=="relative_std"):
            #x2_rel_std=stats["<x^2>"]["std"]/stats["<x^2>"]["mean"]
            #x2_good=x2_rel_std<self.threshold_limit
            D_rel_std=stats["D"]["std"]/stats["D"]["mean"]
            D_good=D_rel_std<self.threshold_limit
            #isconv=x2_good and D_good
            isconv=D_good
            #if self.abs_min_steps>rand_walk.total_steps:
             #   isconv=False
            #if rand_walk.total_steps>5:
            #    isconv=True
            #pos_mean=[]
            #for i in range(self.graph_dim):
                #m,_=self.calc_mean_and_var([rand_walk.particle_pos_pbc_corrected[i]],str(i))
                #pos_mean.append(m)
            #print(rand_walk.total_steps)
            output={"time":rand_walk.total_steps,
                                 "distance":rand_walk.distance,
                                 "D":stats["D_cur"],
                                 "D_mean":stats["D"]["mean"],
                                 #"<x^2>":stats["<x^2>"]["mean"],
                                 #"x2_red_std":x2_rel_std,
                                 "D_rel_std":D_rel_std,
                                 "STD":{
                                     #"<x^2>":stats["<x^2>"]["std"],
                                     "D":stats["D"]["std"]},
                                 "position":[rand_walk.particle_pos_pbc_corrected,rand_walk.particle_location],
                                 #"pomo_pbc":[rand_walk.particle_pos_pbc_corrected[i]%rand_walk.options["latticeObj"].dimensions[i] for i in range(self.graph_dim)],
                                 #"pomo_raw":[rand_walk.particle_location[i]%rand_walk.options["latticeObj"].dimensions[i] for i in range(self.graph_dim)],
                                 "run_id":self.run_id,
                                 "isconv":isconv,
                                 #"pos_mean":pos_mean
                                 }
            self.info.append(output)
            #print(output)
            #if isconv:
            #    raise Exception("STOP")

            self.result_writer.write(output,force=isconv)
            return isconv
        else:
            raise Exception("ERROR:threshold_types!=relative_std has not been implemented. Specified value:"+self.threshold_type)



#The way I want this to work in accordance with the pilot is essentially as follows
#the pilot will spawn a "run" this run then spawns a dimension d "subrun" and a
#dimension d-1 "subrun" - eahc of these subruns are further decmoposed into
#individual calculations of the diffusion coefficent - these are differentialable by their
#runid and dimension - each of these individual claculations are run separately via MPI 
#but calculation of the smae dimension are written to the same file
#then convergence is checked for each dimension
#if convergence is not reached a the process restarts with a new batch number 
#note that run_ids must be continuos accross batches and can not reset
#since they define the random seed
#additionally different batches will be writen to different files to keep files
#small and minimize chances for additional strange behavior with MPI writes
class Overall_Diffusion_Conv():
    def __init__(self,threshold=global_default_thresh,threshold_type="relative_std",usefile=True):
        self.threshold=threshold
        self.threshold_type=threshold_type
        #self.running_stats={"<x^2>":Welford(),"D":Welford(),"t":Welford()}
        self.running_stats={"D":Welford()}
        self.usefile=usefile
        
    def check_conv(self,data,verbose=True):
        if(self.threshold_type!="relative_std"):
            raise Exception("ERROR:threshold_types!=relative_std has not been implemented. Specified value:"+self.threshold_type)
        for i in range(len(data)):
            if(self.usefile):
                reader=utils.ResultReader(data[i])
                #getting these final states may be slightly... faster in reverse
                #or maybe the writer should output something when a final state is reached
                for line in reader.read():
                    if(line["isconv"]):
                        for k in self.running_stats:
                            self.running_stats[k].add(np.array(line[k]))
            else:
                for k in self.running_stats:
                    self.running_stats[k].add(np.array(data[i][k]))
            
            is_all_conv=True         
            rel_stds={}
            means={}
            abs_std={}
            for k in self.running_stats:
                means[k]=self.running_stats[k].mean.item()
                abs_std[k]=self.running_stats[k].var_s.item()
                rel_stds[k]=abs_std[k]/means[k]
                is_all_conv=is_all_conv and rel_stds[k]<self.threshold
                
            if(not verbose):
                return is_all_conv
            else:
                #return is_all_conv,rel_stds
                return is_all_conv,{"rel_stds":rel_stds,"means":means,"abs_std":abs_std}
                    
            
        

#rolling averages convergence in ratio of two things
def old_ratio_conv(past,fn,conv_thresh=global_default_thresh):
    thresh=conv_thresh
    #log("checking conv")
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
            #log("LOGGING STRANGNESS")
            #log("Strange Ratio: "+str(datadict[i])+" "+str(datapair)+"PARAMS:"+json.dumps(list(sys.argv)))
            raise Exception("Invalid Ratio")
            #raise Exception("Strange Ratio: "+str(datadict[i])+" "+str(datapair))
    datalist=[datadict[i] for i in keylist]
    tlist=[0]
    for i in range(1,len(datalist)):
        tlist.append((tlist[i-1]*(i)+datalist[i])/(i+1))
    res=[]
    for i in range(1,len(tlist)):
        res.append(abs(tlist[i]-tlist[i-1]))
    return (sum(res[-10:])/10)<thresh,"dep"
