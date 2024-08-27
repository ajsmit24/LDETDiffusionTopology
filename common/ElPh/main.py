import json
from pathlib import Path
from molecules import Molecules
import numpy as np
import math
from sklearn.model_selection import ParameterGrid
import os

import matplotlib.pyplot as plt
import matplotlib

hbar=6.582119569e-16
#transfer_integral=0.058
transfer_integral=0.1

def write_lattice_file():
   """Write the lattice parameters json file
   """
   lattice = {'nmuc':2,
              'coordmol':[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
              'unitcell':[[1.0, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.0]],
              'supercell':[16, 16, 1],
              'unique':6,
              'uniqinter':[[1, 1, 1, 0, 0, 1], 
              [2, 2, 1, 0, 0, 1], 
              [1, 2, 0, 0, 0, 3], 
              [2, 1, 1, 0, 0, 2], 
              [2, 1, 0, 1, 0, 2], 
              [2, 1, 1, 1, 0, 3]]
   }
   with open('lattice.json', 'w', encoding='utf-8') as f:
      json.dump(lattice, f, ensure_ascii=False, indent=4)

def write_params_file():
   params = {'javg':[0.058, 0.058, 0.058],
             'sigma':[0.029, 0.029, 0.029],
             'nrepeat':50,
             "iseed":3987187,
             'invtau':0.005,
             'temp':0.025
   }
   with open('params.json', 'w', encoding='utf-8') as f:
      json.dump(params, f, ensure_ascii=False, indent=4)

def main(args=None):
   import argparse

   description = "Transient Localization Theory command line interface"

   example_text = """
   examples:

   Calculate charge mobility with:
      elph --mobility
   """

   formatter = argparse.RawDescriptionHelpFormatter
   parser = argparse.ArgumentParser(description=description,
                                    epilog=example_text, 
                                    formatter_class=formatter)

   help = """
   All calculations require a lattice JSON 
   file with the following properties:

   lattice.json:
      nmuc: 
      coordmol: 
      unitcell: 
      supercell: 
      unique:
      uniqinter: 

   """
   parser.add_argument('--lattice_file', nargs='*', help=help,
                        default='lattice', type=str)

   help = """
   All calculations require a params json 
   file with the following properties:

   params.json:
      javg: 
      sigma: 
      nrepeat:
      iseed: 
      invtau:
      temp:

   """
   parser.add_argument('--params_file', nargs='*', help=help,
                        default='params', type=str)

   help = ("write example of lattice and params files")
   parser.add_argument('--write_files', action='store_true' , help=help)

   help = ("Calculate charge mobility")
   parser.add_argument('--mobility', action='store_true' , help=help)

   args = parser.parse_args(args)

   if args.write_files:
      write_lattice_file()
      write_params_file()
      return

   print('Initializing ElPh')

   if args.mobility:
      if not Path(args.lattice_file + '.json').is_file():
         msg = 'Lattice file could not be found'
         raise FileNotFoundError(msg)

      if not Path(args.params_file + '.json').is_file():
         msg = 'Params file could not be found'
         raise FileNotFoundError(msg)

      mols = Molecules(lattice_file=args.lattice_file, 
      params_file=args.params_file)
         
      mobx, moby = mols.get_mobility()
      with open('results.json', 'w', encoding='utf-8') as f:
         json.dump(mols.results, f)
         
def test_calcs():
    write_lattice_file()
    write_params_file()
    mols = Molecules(lattice_file='lattice', 
    params_file='params')
    mobx, moby = mols.get_mobility()
    with open('results.json', 'w', encoding='utf-8') as f:
       json.dump(mols.results, f)

def K_to_eV(T):
    kB=8.617333262e-5
    return kB*T

def res_write(res):
    f=open("res.out","a")
    f.write(json.dumps(res)+"\n")
    f.close()

def write_T_dep_params(T,fn='params.json',seed=3987187):
    T_eV=K_to_eV(T)
    #300K dynamic disorder
    base_dynamic_rel=0.029/0.058
    base_temp=0.025
    base_dynamic=base_dynamic_rel*transfer_integral
    #sqrt propotionality constant
    prop=0.029/math.sqrt(base_temp)
    params = {'javg':[transfer_integral]*3,
              'sigma':[prop*math.sqrt(T_eV)]*3,
              'nrepeat':25,
              "iseed":seed,
              'invtau':0.005,
              'temp':T_eV
    }
    with open(fn, 'w', encoding='utf-8') as f:
       json.dump(params, f, ensure_ascii=False, indent=4)
       
def read_std_res():
    f=open('results.json','r')
    data=json.loads(f.read())
    f.close()
    return (data["mobx"]+data["moby"])/2,sum(data["squared_length"])/len(data["squared_length"])


def calc_T_point(p):
    T=p["temp"]
    s=p["static"]
    fn="params_"+str(T)+"_"+str(s)
    write_T_dep_params(T,fn=fn+".json",seed=p["seed"])
    mols = Molecules(lattice_file='lattice',params_file=fn,static_disorder_params={"abs":s})
    mobx, moby = mols.get_mobility()
    data=mols.results
    mob,sqlen=((data["mobx"]+data["moby"])/2,sum(data["squared_length"])/len(data["squared_length"]))
    os.remove(fn+".json")
    f=open("log.log","a")
    f.write("DONE "+str(T)+", "+str(s)+","+str(p["s_true"])+"\n")
    f.close()
    res_write([T,mob,sqlen,s,p["s_true"]])

def get_rand_static(s):
    return np.random.normal(0,1)*s

def gen_temp_dep_plot(minT=10,maxT=1000,nT=90,useMPI=False,staticmin=0,staticmax=10,staticn=10,staticreps=5):
    temp_range=np.linspace(minT,maxT,nT)
    static_range=np.linspace(staticmin,staticmax,staticn)
    params=ParameterGrid({"static":static_range,"temp":temp_range})
    params_list=[]
    for p in params:
       for i in range(staticreps):
          s=p["static"]*transfer_integral
          if(staticreps>1):
             s=get_rand_static(s)
          params_list.append({"temp":p["temp"],"seed":len(params_list),"static":s,"s_true":p["static"]})
    params=params_list
    write_lattice_file()
    result=[]
    if(useMPI):
       from mpi4py.futures import MPIPoolExecutor
       with MPIPoolExecutor() as pool:
            result = pool.map(calc_T_point, params)
    else:
       for p in params:
           result.append(calc_T_point(p))
    result=[r for r in result]
    f=open("log.log","a")
    f.write("HERE\n")
    f.close()
    f=open("log.log","a")
    f.write(str(type(result)))
    f.close()
    f=open("points.json","w+")
    f.write(json.dumps({"points":result}))
    f.close()
    
def get_expo_fit(xvar,yvar):
    maxdex=yvar.index(max(yvar))
    coef=np.polyfit([-1*(1/xvar[i]) for i in range(maxdex)], np.log([yvar[i] for i in range(maxdex)]), 1)
    def expofit(x):
        return [math.exp(coef[1]) * math.exp(coef[0] * (-1/xi)) for xi in x]
    return expofit,coef,maxdex
def vis(do_static=True,norm_mob=True):
    f=open("points.json","r")
    data=json.loads(f.read())
    f.close()
    static={}
    data["points"]=[p for p in data["points"]]
    if(do_static):
        for p in data["points"]:
                if(p[4] not in static):
                    static[p[4]]={}
                if(p[0] not in static[p[4]]):
                    static[p[4]][p[0]]={"mob":[],"loc":[]}
                static[p[4]][p[0]]["mob"].append(p[1])
                static[p[4]][p[0]]["loc"].append(p[2])
        new_static={}
        for s in static:
            new_static[s]=[]
            for t in static[s]:
                mob=static[s][t]["mob"]
                loc=static[s][t]["loc"]
                new_static[s].append([t,sum(mob)/len(mob),sum(loc)/len(loc),s])
        static=new_static
    else:
        #static[0]=[p for p in data["points"] if(p[3]<0.01 and p[3]>-0.001)]
        static[0]=[p for p in data["points"] if(p[3]>0.95 and p[3]<1.05)]
    
    for s in static:
        if(s<1):
            continue
        points=static[s]
        points=[p for p in points if(not np.isnan(p[1]))]
        if(len(points)<1):
            print("NO DATA FOR ",s)
            continue
        fig, ax1 = plt.subplots()
        normfactor=1
        if(norm_mob):
            diff=[abs(100-p[0]) for p in points]
            normfactor=points[diff.index(min(diff))][1]
        color = 'tab:red'
        #ax1.scatter([p[0] for p in points],[p[1] for p in points],color=color)
        xvar=[K_to_eV(p[0])/transfer_integral for p in points]
        yvar=[p[1]/normfactor for p in points]
        ax1.scatter(xvar,yvar,color=color)
        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.scatter(xvar,[p[2] for p in points],color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        expofit,coef,maxdex=get_expo_fit(xvar,yvar)
        print(maxdex)
        ax1.plot(xvar, expofit(xvar),'--g', label="Fitted Curve")
        ax1.scatter([xvar[i] for i in range(maxdex)], [yvar[i] for i in range(maxdex)],color="tab:green")
        print(coef[0],math.exp(coef[1]))
        print(coef[0]*transfer_integral)

        if(do_static):
            #plt.colorbar()
            plt.title("static disorder: "+str(s))
        else:
            plt.title("Mobility vs Temperature for Highly Pure Synthetic Organics")
        plt.xlabel('Temperature/J')
        plt.ylabel('Mobility')
        plt.show()
        plt.figure()
        
def plot_activation_energy():
    f=open("points.json","r")
    data=json.loads(f.read())
    f.close()
    static={}
    tempdata={}
    data["points"]=[p for p in data["points"]]
    for p in data["points"]:
            if(p[4] not in static):
                static[p[4]]={}
            if(p[0] not in static[p[4]]):
                static[p[4]][p[0]]={"mob":[],"loc":[]}
            static[p[4]][p[0]]["mob"].append(p[1])
            static[p[4]][p[0]]["loc"].append(p[2])
    new_static={}
    for s in static:
        new_static[s]=[]
        for t in static[s]:
            mob=static[s][t]["mob"]
            loc=static[s][t]["loc"]
            new_static[s].append([t,sum(mob)/len(mob),sum(loc)/len(loc),s])
    static=new_static
    plotpoints=[]
    for s in static:
        slist=[p for p in static[s] if(not np.isnan(p[1]))]
        fitx=[K_to_eV(p[0])/transfer_integral for p in slist]
        diff=[abs(100-p[0]) for p in slist]
        normfactor=slist[diff.index(min(diff))][1]
        fity=[p[1]/normfactor for p in slist]
        expofit,coef,maxdex=get_expo_fit(fitx,fity)
        plotpoints.append([s,coef[0]*transfer_integral])
    x=[p[0] for p in plotpoints]
    y=[p[1]  for p in plotpoints]
    plt.scatter(x,y)
    plt.figure()
        
def vis_static_max():
    f=open("points.json","r")
    data=json.loads(f.read())
    f.close()
    static={}
    data["points"]=[p for p in data["points"]]
    for p in data["points"]:
            print(p)
            if(p[4] not in static):
                static[p[4]]={}
            if(p[0] not in static[p[4]]):
                static[p[4]][p[0]]={"mob":[],"loc":[]}
            static[p[4]][p[0]]["mob"].append(p[1])
            static[p[4]][p[0]]["loc"].append(p[2])
    new_static={}
    for s in static:
        new_static[s]=[]
        for t in static[s]:
            mob=static[s][t]["mob"]
            loc=static[s][t]["loc"]
            new_static[s].append([t,sum(mob)/len(mob),sum(loc)/len(loc),s])
    static=new_static
    maxpoints=[]
    for s in static:
        slist=[p for p in static[s] if(not np.isnan(p[1]))]
        maxmob=max([p[1] for p in slist])
        maxmobdex=[p[1] for p in slist].index(maxmob)
        maxpoints.append([s,slist[maxmobdex][0]])
    x=[p[0] for p in maxpoints]
    y=[K_to_eV(p[1])/transfer_integral  for p in maxpoints]
    #plt.scatter(x,y)
    plt.title("Transition Temperature vs Static Disorder")
    plt.xlabel('% static contribution to disorder*')
    plt.ylabel('Transition temperature/J**')
    """plt.text(0,-10,'*Negative static disorder is not realizabile in real\n '+
             'systems but demonstrates the trend from a theory stand point',
             ha='center')
    plt.text(0,-17.5,'**Minimum allowed temperature was 10K due to numeric '+
             'limitations.\n Points at 10K may represent'+
             ' lower temperature transitions.',
             ha='center')"""
    coef = np.polyfit(x,y,1)
    m,b = coef
    poly1d_fn = np.poly1d(coef) 
    plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k',label=f'$y = {m:.1f}x {b:+.1f}$')
    plt.legend()
    
    
if __name__  == '__main__':
   #main()
   vis()
   plot_activation_energy()
   vis_static_max()
   #vis(do_static=False)
   #test_calcs()
   #gen_temp_dep_plot(useMPI=True)
   #gen_temp_dep_plot(useMPI=False)
