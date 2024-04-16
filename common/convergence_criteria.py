# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:50:41 2024

@author: smith
"""

import json
import sys

import numpy as np

from welford import Welford
#https://github.com/a-mitani/welford

import utils
mylogger=utils.mlogger("main.log")
log=mylogger.log


class Diffusion_Conv():
    def __init__(self,result_writer,graph_dim,threshold_type="relative_std",threshold_limit=0.05):
        self.threshold_types=["relative_std","absolute_value"]
        if(threshold_type not in self.threshold_types):
            raise Exception("ERROR: Invalid argument value threshold_type="+threshold_type)
        self.threshold_type=threshold_type
        self.threshold_limit=threshold_limit
        self.graph_dim=graph_dim
        self.result_writer=result_writer
        
        self.conv_check_count=0
        
        self.running_stats={"<x^2>":Welford(),"D":Welford()}
        
    def calc_mean_and_var(self,new_data,stat_type):
        for nd in new_data:
            self.running_stats[stat_type].add(np.array(nd))
        return self.running_stats[stat_type].mean, self.running_stats[stat_type].var_s
    def update_diffusion_coef(self,time,distance):
        #msd=mean squared displacement
        msd_x2,std_x2=self.calc_mean_and_var([distance],"<x^2>")
        mean_D,std_D=self.calc_mean_and_var([msd_x2/((2**self.graph_dim)*time)],"D")
        return {"<x^2>":{"mean":msd_x2,"std":std_x2},"D":{"mean":mean_D,"std":std_D}}
        
    def check_conv(self,rand_walk):
        stats=self.update_diffusion_coef(rand_walk.total_steps,rand_walk.distance)
        if(self.threshold_types=="relative_std"):
            x2_good=stats["<x^2>"]["std"]/stats["<x^2>"]["mean"]<self.threshold_limit
            D_good=stats["D"]["std"]/stats["D"]["mean"]<self.threshold_limit
            self.result_writer.write({"time":rand_walk.total_steps,
                                 "distance":rand_walk.distance,
                                 "D":stats["D"]["mean"],
                                 "<x^2>":stats["<x^2>"]["mean"],
                                 "STD":{"<x^2>":stats["<x^2>"]["std"],"D":stats["D"]["std"]},
                                 "position":self.particle_pos_pbc_corrected
                                 })
            return x2_good and D_good
        else:
            raise Exception("ERROR:threshold_types!=relative_std has not been implemented")




#rolling averages convergence in ratio of two things
def old_ratio_conv(past,fn,conv_thresh=0.001):
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