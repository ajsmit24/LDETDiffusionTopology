# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:00:30 2024

@author: smith
"""

#plotly
import json
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import pandas as pd    
import numpy as np
import statistics
from tabulate import tabulate
import colormapshifter
import math
import statistics

matplotlib.rcParams['figure.dpi'] = 600

 
font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 11}
matplotlib.rc('font', **font)

#fn="mparams_67500.json"
#fn="mparams_8000.json"
#fn="mparams_31250.json"
#fn="mparams_225000.json"
#fn="mparams_720000.json"
fn="mparams.json"

#plotter="mpl"
plotter="pltly"


f=open(fn,"r")
data=json.loads(f.read())
f.close()


expdata=[0.09,0.11,0.27,0.27]
mu_stdev=statistics.stdev(expdata)
av_mu=sum(expdata)/len(expdata)

cutoff=np.log10(10000000000000)
#chain_err_adj=1.4
#chain_err_adj=1.24*2.61
#chain_err_adj=1.6*2*0.1
chain_err_adj=1.6*2
def filter1(x):
    return abs(x)<cutoff

def filter2(x,y):
    return abs(x)<cutoff and y>chain_err_adj

def thresh_count(data,t):
    return len([d for d in data if(abs(d)<t)])

def find_rH_given_lambda(lmbda):
    hbar=1.054571817e-34
    kb_eV=8.617333262e-5
    kB=1.380649e-23 #J/K
    fund_charge=1.602176e-19
    eV_to_J=1.60218e-19
    nm_to_m=1e-9
    T=300
    lmbda_si=lmbda*eV_to_J
    cm2_to_m2=1e-4
    av_mu_si=av_mu*cm2_to_m2
    x_si=math.sqrt((chain_err_adj*av_mu_si*hbar/(fund_charge*math.pi*2))*kB*T*
              math.sqrt(4*math.pi*lmbda_si*kB*T)*np.exp(lmbda_si/(4*kB*T)))
    y=(1/chain_err_adj)*(fund_charge/(kB*T))*(1/ math.sqrt(4*math.pi*lmbda_si*kB*T))*np.exp(-1*lmbda_si/(4*kB*T))*((0.01*nm_to_m*eV_to_J)**2)*math.pi*2/hbar
    return x_si/(nm_to_m*eV_to_J)
    

def mobility(p):
    kB=1.380649e-23 #J/K
    fund_charge=1.602176e-19
    k=float(p[3][0])
    r=p[0]
    T=300
    #nm^2 V^-1 s^-1
    mu=fund_charge/(kB*T)*k*r*r
    #cm^2 V^-1 s^-1
    mu=mu/(1e14)
    return mu

def mobility_scaled(p):
    return  mobility(p)/chain_err_adj
def mobility_scaled_diff(p):
    return  abs(mobility(p)/chain_err_adj-av_mu)
def __mobility(k,r):
    kB=1.380649e-23 #J/K
    fund_charge=1.602176e-19
    T=300
    #nm^2 V^-1 s^-1
    mu=fund_charge/(kB*T)*k*r*r
    #cm^2 V^-1 s^-1
    mu=mu/(1e14)
    return mu
def mobility_both(p):
    r=p[0]
    return [__mobility(float(p[3][0]),r),__mobility(float(p[3][1]),r)]

data=[[d[0],d[1],d[2],mobility_both(d),d[4],np.log10(mobility(d))] for d in data]
print("%"*20)
print(len(data))
print("r",min([d[0] for d in data]),max([d[0] for d in data]))
print("H",min([d[2] for d in data]),max([d[2] for d in data]))
print("L",min([d[1] for d in data]),max([d[1] for d in data]))
print("%"*20)
#rnn,lambda,coupling,[calc_rate,req_rate], is calc>req
data = [d for d in data if(d[0]<=2.5 and d[0]>=0.5)]
data = [d for d in data if(d[2]>=0.001 and d[2]<=0.03)]
data = [d for d in data if(d[1]>=0.15 and d[1]<=0.361)]


dr=set([data[i][0]-data[i-1][0] for i in range(len(data))])
dL=set([data[i][1]-data[i-1][1] for i in range(len(data))])
dH=set([data[i][2]-data[i-1][2] for i in range(len(data))])
print("*"*20)
print("dr",dr)
print("dL",dL)
print("dH",dH)
print("*"*20)

x=[d[0] for d in data]
y=[d[1] for d in data]
z=[d[2] for d in data]



xlabel='Inter Cofactor Distance (nm)'
ylabel='Reorganization Energy (eV)'
zlabel='Coupling (eV)'
clabel="|μ{EXP}-μ{CALC}|"
mulabel="Mobility cm^2/Vs"
rlable="μ{CALC}/μ{EXP}"
#rnn,lambda,coupling,[calc_rate,req_rate], is calc>req
pdata=[[data[i][0],data[i][1],data[i][2],data[i][5]] for i in range(len(data))]
df=pd.DataFrame(pdata,columns=[xlabel,ylabel,zlabel,clabel])
fig =px.scatter_3d(df, x=xlabel, y=ylabel, z=zlabel,color=clabel,hover_data=df,
                   #color_continuous_scale="RdYlBu",color_continuous_midpoint =0)
                   color_continuous_scale="plasma")

av_mu=0.185
mu_low=av_mu-mu_stdev*0.5
mu_high=av_mu+mu_stdev*0.5

#d[3]=[calc_mob,req_mob]
isgood=[1 for d in data if(abs(d[3][0]-av_mu)<=0.01)]
print(sum(isgood)/len(data),sum(isgood),len(data))
isgood=[1 for d in data if(abs(d[3][0]-av_mu)<=mu_stdev)]
print(sum(isgood)/len(data),sum(isgood),len(data))

display="desktop"
fn="marcus_3Dplot4.html"
if(display=="mobile"):
    fn=fn.split(".")[0]+"_"+display+"."+fn.split(".")[-1]
    fig.update_traces(marker=dict(size=30))
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=3, y=3, z=3)
    )
    font=dict(
            family="Courier New, monospace",
            size=18,  # Set the font size here
        )
    hoverfont=dict(
            family="Courier New, monospace",
            size=24,  # Set the font size here
        )
    title="Theoretical Parameters that give Mobilities Within<br> Experimental Precision (0.01cm^2/Vs)<br> of the Experimental Average"
    fig.update_layout(scene_camera=camera, title=title, font=font,hoverlabel=dict(font=hoverfont))
fig.write_html(fn)
fig.show()
#plt.xticks(cutoffs, cutoffs)
params={"Inter-cofactor Distance":[pd[0] for pd in pdata],
        "Reorganization Energy":[pd[1] for pd in pdata],
        "Coupling":[pd[2] for pd in pdata],
        }
table={"Parameter":[],"Minimum":[],"Maximum":[]}
units={"Inter-cofactor Distance":"(nm)","Reorganization Energy":"(eV)","Coupling":"(meV)"}
for p in params:
    table["Parameter"].append(p+units[p])
    table["Minimum"].append(round(min(params[p]),5))
    table["Maximum"].append(round(max(params[p]),5))
print(tabulate(table, headers="keys"))


plt.figure()
plt.scatter([d[1] for d in data],[d[0]*d[2] for d in data],c=[d[5] for d in data])
plt.scatter([d[1] for d in data if(abs(abs(d[5]))<0.01)],[d[0]*d[2] for d in data if(abs(abs(d[5]))<0.01)], color="black")

plt.figure()
ax = plt.gca()

lmbda_rng=np.linspace(0.16,0.36,50)
R_vals=[0.75,1,1.5,2,2.5]
#R_vals=[1.25]

colorsets=[["tab:blue","cornflowerblue"],
           ["tab:orange","sandybrown"],
           ["tab:green","olivedrab"],
           ["tab:red","salmon"],
           ["tab:purple","thistle"]
           ]
hatches=["X","|","-","X","|","-","X","|"]
hatches=["X","|","-","O","*","."]
i=0
for R in R_vals:
    #adjust mu av because used in find_rH_given_lambda
    #here we are doing +/- one std
    av_mu=0.185
    #factor of 1000 unit conversion to meV
    plt.plot(lmbda_rng,[(find_rH_given_lambda(l)/R)*1000 for l in lmbda_rng],label=str(R),color=colorsets[i][0])
    av_mu=mu_low
    y1=[(find_rH_given_lambda(l)/R)*1000 for l in lmbda_rng]
    plt.plot(lmbda_rng,y1,'--',color=colorsets[i][-1])
    av_mu=mu_high
    y2=[(find_rH_given_lambda(l)/R)*1000 for l in lmbda_rng]
    plt.plot(lmbda_rng,y2,'--',color=colorsets[i][-1])
    #if(i%2<7):
    plt.fill(np.append(lmbda_rng, lmbda_rng[::-1]), np.append(y1, y2[::-1]), colorsets[i][-1],alpha=0.4)
        #plt.fill_between(np.append(lmbda_rng, lmbda_rng[::-1]), np.append(y1, y2[::-1]),
     #                facecolor="none", hatch=hatches[i], edgecolor=colorsets[i][-1],
     #                linewidth=0.0,alpha=0.85)

    i+=1
#plt.plot([0.16,0.175,0.25,0.325,0.35,0.36],[16,17.5,25,32.5,35,36],linestyle='dashdot',color="black")
plt.xlabel("Reorganization Energy (eV)")
plt.ylabel("Coupling (meV)")
plt.title("Viable Couplings and Reorganization Energies at\n Various Inter-cofactor Distances")
plt.legend(title="Cofactor\n Center-to-Center\n Distance(nm)",)
plt.tick_params(which='minor', length=5, width=0.25) 
#plt.yticks(ticks=[i for i in range(50) if(i%10==0)],minor=False)
plt.minorticks_on()
#plt.xticks([0.15,0.20,0.25,0.30,0.35])
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
plt.grid(visible=True,which = 'minor', alpha = 0.25)
plt.grid(which = 'major', alpha = 0.75)
plt.savefig('rainbow_good.png', dpi=300)


    