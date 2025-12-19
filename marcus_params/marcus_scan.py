# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:39:55 2023

@author: smith
"""

import sys

#fundamental constants
kB=1.380649e-23 #J/K
fund_charge=1.602176e-19 #fundamental charge of a proton in Coulomb

#expiremental constants
#https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-76671-5/MediaObjects/41598_2020_76671_MOESM1_ESM.pdf
#cm^2 V^-1 s^-1
mu=(0.27+0.11)/2
#m^2 V^-1 s^-1
mu=mu/(100**2)

#fixed parameters
T=300
delG=0
hbar=1.054571817e-34
import math


#variable parameters
v_params={
    "Rnn":[0.1,10],#nm
    "Coupling":[0.001,0.1],#eV
    "lambda":[0.16,0.36]#eV
          
}
unit_convert={
    "Rnn":1e-9,
    "Coupling":1.60218e-19,
    "lambda":1.60218e-19,
}

for k,v in v_params.items():
  v_params[k]=[p*unit_convert[k] for p in v_params[k]]

def calc_req_hopping(mu,r,T):
  return (kB*T*mu)/(fund_charge*r*r)

def calc_hop_from_params(lmbd,coupling):
  return (2*math.pi/hbar)*(1/math.sqrt(4*math.pi*lmbd*kB*T))*coupling*coupling*math.exp(-1*((delG+lmbd)**2)/(4*lmbd*kB*T))
  
import numpy as np

linesteps=100
try:
    linesteps=int(sys.argv[-1])
except Exception as e:
    print(e,"setting linesteps=10")
    linesteps=10

linespaces={}
for k in v_params:
    linespaces[k]=np.linspace(v_params[k][0],v_params[k][1],linesteps)
    
points=[]
outs=[]

klist=[k for k in v_params]

Rnn_dex=klist.index("Rnn")
lambda_dex=klist.index("lambda")
Coupling_dex=klist.index("Coupling")

for i in range(len(linespaces[klist[Rnn_dex]])):
    for j in range(len(linespaces[klist[lambda_dex]])):
        for k in range(len(linespaces[klist[Coupling_dex]])):
            reqrate=calc_req_hopping(mu,linespaces["Rnn"][i],T)
            crate=calc_hop_from_params(linespaces["lambda"][j],linespaces["Coupling"][k])
            hops=["{:e}".format(crate),"{:e}".format(reqrate)]
            points.append([
                linespaces[klist[Rnn_dex]][i]/unit_convert[klist[Rnn_dex]],
                linespaces[klist[lambda_dex]][j]/unit_convert[klist[lambda_dex]],
                linespaces[klist[Coupling_dex]][k]/unit_convert[klist[Coupling_dex]],
                hops,
                int(crate>reqrate)
                ])

import json

f=open("mparams.json","w+")
f.write(json.dumps(points))
f.close()

    

