import json
from pathlib import Path
from molecules import Molecules
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib

hbar=6.582119569e-16

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
def write_T_dep_params(T):
    T_eV=K_to_eV(T)
    #sqrt propotionality constant
    prop=0.029/math.sqrt(0.025)
    params = {'javg':[0.058, 0.058, 0.058],
              'sigma':[prop*math.sqrt(T_eV)]*3,
              #'sigma':[0.029]*3,
              'nrepeat':50,
              "iseed":3987187,
              'invtau':0.005,
              'temp':T_eV
    }
    with open('params.json', 'w', encoding='utf-8') as f:
       json.dump(params, f, ensure_ascii=False, indent=4)
       
def read_std_res():
    f=open('results.json','r')
    data=json.loads(f.read())
    f.close()
    return (data["mobx"]+data["moby"])/2,sum(data["squared_length"])/len(data["squared_length"])
def gen_temp_dep_plot(minT=10,maxT=500,nT=10):
    temp_range=np.linspace(minT,maxT,nT)
    write_lattice_file()
    points=[]
    for T in temp_range:
        write_T_dep_params(T)
        mols = Molecules(lattice_file='lattice',params_file='params')
        mobx, moby = mols.get_mobility()
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(mols.results, f)
        mob,sqlen=read_std_res()
        points.append([T,mob])
    #plt.scatter([p[0] for p in points],[p[1] for p in points])
    #plt.show()
    #plt.figure()
    f=open("points.json","w+")
    f.write(json.dumps({"points":points}))
    f.close()
def vis(do_static=True):
    f=open("points.json","r")
    data=json.loads(f.read())
    f.close()
    static={}
    data["points"]=[p for p in data["points"] if(p[3]<0.4)]
    if(do_static):
        for p in data["points"]:
            if(p[3] not in static):
                static[p[3]]=[]
            static[p[3]].append(p)
    else:
        static[0]=[p for p in data["points"] if(p[3]<0.01 and p[3]>-0.001)]
    
    for s in static:
        points=static[s]
        points=[p for p in points]
        fig, ax1 = plt.subplots()
        
        plt.scatter([p[0] for p in points],[p[1] for p in points],c=[p[3] for p in points])
        #color = 'tab:blue'
        #ax2 = ax1.twinx()
        #ax2.scatter([p[0] for p in points],[p[2] for p in points],color=color)
        #ax2.scatter([p[0] for p in points],[p[2] for p in points],color=color)
        #ax2.tick_params(axis='y', labelcolor=color)
        if(do_static):
            plt.colorbar()
            plt.title("static disorder: "+str(s))
        else:
            plt.title("Mobility vs Temperature for Highly Pure Synthetic Organics")
        plt.xlabel('Temperature (K)')
        plt.ylabel('Mobility')
        plt.show()
        plt.figure()
def vis_static_max():
    f=open("points.json","r")
    data=json.loads(f.read())
    f.close()
    static={}
    data["points"]=[p for p in data["points"]]
    for p in data["points"]:
            if(p[3] not in static):
                static[p[3]]=[]
            static[p[3]].append(p)
    maxpoints=[]
    for s in static:
        maxmob=max([p[1] for p in static[s]])
        maxmobdex=[p[1] for p in static[s]].index(maxmob)
        maxpoints.append([s,static[s][maxmobdex][0]])
    plt.scatter([p[0] for p in maxpoints],[p[1] for p in maxpoints])
    plt.title("Transition Temperature vs Static Disorder")
    plt.xlabel('% static contribution to disorder*')
    plt.ylabel('Transition temperature**')
    plt.text(0,-10,'*Negative static disorder is not realizabile in real\n '+
             'systems but demonstrates the trend from a theory stand point',
             ha='center')
    plt.text(0,-17.5,'**Minimum allowed temperature was 10K due to numeric '+
             'limitations.\n Points at 10K may represent'+
             ' lower temperature transitions.',
             ha='center')
if __name__  == '__main__':
   #main()
   #gen_temp_dep_plot()
   #vis()
   vis_static_max()
   vis(do_static=False)
   #test_calcs()
