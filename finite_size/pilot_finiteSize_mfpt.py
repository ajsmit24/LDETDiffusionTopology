# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:14:24 2024

@author: smith
"""

import sys
sys.path.insert(0, '../common/')
sys.path.insert(0, '/home/ajs193/cable_bacteria/diffusion_topology/common/')
import integer_dimension_lattice as idl
import utils
import convergence_criteria as convcrit
import random_walker
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for lattice diffusion simulation")

    # Number of lattice points option
    parser.add_argument('-n', '--num_points', type=int, required=True,
                        help='Number of lattice points')

    # Highest dimension option
    parser.add_argument('-d', '--dimension', type=int, required=True,
                        help='Highest dimension (e.g., 1 for 1D, 2 for 2D, etc.)')

    # Cutoff distance option
    parser.add_argument('-c', '--cutoff', type=float, required=True,
                        help='Cutoff distance for interaction')

    # Job name option
    parser.add_argument('-j', '--job_name', type=str, required=True,
                        help='Job name')
    parser.add_argument('-fdl','--finite_dim_len', type=int,required=True, help="the length of finite dimensions")

    args = parser.parse_args()

    return args

reslogger,reslog,jobstem,args=(None,None,None,None)
if(__name__=="__main__"):
 args=parse_arguments()
 jobstem=args.job_name+"_maxd-"+str(args.dimension)+"_np-"+str(args.num_points)+"_fdl-"+str(args.finite_dim_len)
 reslogger=utils.mlogger(jobstem+".log")
 reslog=reslogger.log


output_files={}
batch_output_file_format="{jobname}_mfptunidir_d{dim}_b{batch}_x{xcut}.out"

def single_first_passage_calc(params):
    random.seed(params["run_id"])
    dim_num=params["d"]
    peroidic_size=params["peroidic_unit_size"]
    surface_boundary=params["x_cut"]
    lattice_dims=[params["fdl"]]*dim_num
    lattice_dims[0]=peroidic_size
    peroidic_dims=[False]*dim_num
    peroidic_dims[0]=True
    lattice=idl.UniformNDLatticeConstructor(dims=lattice_dims,periodic=peroidic_dims)
    options={"latticeObj":lattice,"graphType":"lattice"}
    endcriteria={
        "uni_dir_surface":{
            "bound_dist":surface_boundary,"writer":params["result_writer"],
            "bound_dim":lattice.reorientation[0],"reflecting":True}
        }
    rand_walker=random_walker.RandomWalker(lattice.graph,particle_initial_pos=lattice.particle_position,
                                           endcriteria=endcriteria,options=options)
    rand_walker.run_random_walk()
    return {"mft":rand_walker.total_steps,"dim":dim_num}

def run_dim_batch(dim,job_options,batchnumber=1,lastjobid=0):
    global output_files
    joblist=[]
    btch_fn=batch_output_file_format.replace(
        "{jobname}",job_options["job_name"]).replace(
            "{dim}",str(dim)).replace("{batch}",str(batchnumber)).replace("{xcut}",str(job_options["x_cut"]))
    res_writer=utils.ResultWriter(btch_fn,frequency=job_options["write_freq"],mute=True)
    for i in range(job_options["calcs_per_batch"]):
        joblist.append({
            "peroidic_unit_size":job_options["peroidic_unit_size"],
            "run_id":lastjobid,
            "d":dim,
            "x_cut":job_options["x_cut"],
            "result_writer":res_writer,
            "fdl":job_options["fdl"]
            })
        if(dim not in output_files):
            output_files[dim]=set()
        output_files[dim].add(btch_fn)
        lastjobid+=1
    return joblist,lastjobid

def run_job(job_name,finite_dim_len,x_cut,highest_dimension,calcs_per_batch,peroidic_unit_size,write_freq=25,useMPI=False,d3AutoTrans=True):
    global output_files
    job_options={
        "calcs_per_batch":calcs_per_batch,
        "peroidic_unit_size":peroidic_unit_size,
        "write_freq":write_freq,
        "job_name":job_name,
        "x_cut":x_cut,
        "fdl":finite_dim_len
        }
    #dimension offset used to transition between dimensions 
    dofset=1
    #previous system dident allow for jumps of more than 1 dimensions ie 3D to 1D
    #so it used for loops with a counter subtracted off from the highest dim
    #here we supply those subractive values manually
    dims_pos_subtractive=[0,1]
    #incase of dimension 3 and this transition parameter
    #instead of comparing the highest and highets-dofset(1) 
    #dimensions, compare 3 and 1
    #this is the CB relivant comparison
    if(d3AutoTrans and highest_dimension==3):
       dofset=2
       dims_pos_subtractive=[0,2]

    lastrunid={highest_dimension:0,highest_dimension-dofset:0}
    is_total_conv=False
    total_conv_checker={highest_dimension:convcrit.Rolling_Av_Conv(["mft"],usefile=False),
                        highest_dimension-dofset:convcrit.Rolling_Av_Conv(["mft"],usefile=False)}
    loop_count=0
    conv_by_dim={highest_dimension:False,highest_dimension-dofset:False}
    prev_data={"relstd":{highest_dimension:1e99,highest_dimension-dofset:1e99},"conv":{highest_dimension:False,highest_dimension-dofset:False}}
    while(not is_total_conv):
        mpi_job_list=[]
        for i in dims_pos_subtractive:
            if(not conv_by_dim[highest_dimension-i]):
                jlist,lastid=run_dim_batch(highest_dimension-i,job_options,batchnumber=loop_count,lastjobid=lastrunid[highest_dimension-i])
                mpi_job_list+=jlist
                lastrunid[highest_dimension-i]=lastid
        result=[]
        if(useMPI):
            from mpi4py.futures import MPIPoolExecutor
            with MPIPoolExecutor() as pool:
                result = pool.map(single_first_passage_calc, mpi_job_list)
        else:
            for jp in mpi_job_list:
                result.append(single_first_passage_calc(jp))
        cleaned_res={highest_dimension:[],highest_dimension-dofset:[]}
        for res in result:
            cleaned_res[res["dim"]].append(res)
        temp_cov=True
        rel_std_list={}
        for i in dims_pos_subtractive:
            if(len(cleaned_res[highest_dimension-i])<1):
                rel_stds=prev_data["relstd"][highest_dimension-i]
                conv=prev_data["conv"][highest_dimension-i]
            else:
                conv,rel_stds=total_conv_checker[highest_dimension-i].check_conv(cleaned_res[highest_dimension-i])
                prev_data["conv"][highest_dimension-i]=conv
                prev_data["relstd"][highest_dimension-i]=rel_stds
            rel_std_list[highest_dimension-i]=rel_stds
            conv_by_dim[highest_dimension-i]=conv
            temp_cov=temp_cov and conv_by_dim[highest_dimension-i]
        is_total_conv=temp_cov
        print(rel_std_list)
        reslog([is_total_conv,conv_by_dim,loop_count,rel_std_list])
        loop_count+=1

if(__name__=="__main__"):                
	run_job(jobstem,args.finite_dim_len,args.cutoff,args.dimension,3,args.num_points,useMPI=True)
