# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:50:41 2024

@author: smith
"""

import json
import sys

import utils
mylogger=utils.mlogger("main.log")
log=mylogger.log


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