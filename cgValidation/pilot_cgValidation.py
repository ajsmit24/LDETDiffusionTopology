import sys
sys.path.insert(0, '../common/')
import integer_dimension_lattice as idl
import utils
import convergence_criteria as convcrit
import random_walker
import random

reslogger=utils.mlogger("res_dif.log")
reslog=reslogger.log


output_files={}
batch_output_file_format="{jobname}_diffusion_d{dim}_b{batch}.out"

def single_diffusion_calc(params):
    random.seed(params["run_id"])
    dim_num=params["d"]
    peroidic_size=params["peroidic_unit_size"]
    lattice=idl.UniformNDLatticeConstructor(dims=[peroidic_size]*dim_num,periodic=[True]*dim_num)
    conv_checker=convcrit.Diffusion_Conv(params["run_id"], params["result_writer"], dim_num)
    options={"latticeObj":lattice,"graphType":"lattice"}
    rand_walker=random_walker.RandomWalker(lattice.graph,particle_initial_pos=lattice.particle_position,
                                           endcriteria={"conv_class":conv_checker},options=options)
    rand_walker.run_random_walk()
    return None

def run_dim_batch(dim,job_options,batchnumber=1,lastjobid=0):
    global output_files
    joblist=[]
    btch_fn=batch_output_file_format.replace(
        "{jobname}",job_options["job_name"]).replace(
            "{dim}",str(dim)).replace("{batch}",str(batchnumber))
    res_writer=utils.ResultWriter(btch_fn,frequency=job_options["write_freq"])
    for i in range(job_options["calcs_per_batch"]):
        joblist.append({
            "peroidic_unit_size":job_options["peroidic_unit_size"],
            "run_id":lastjobid,
            "d":dim,
            "result_writer":res_writer
            })
        if(dim not in output_files):
            output_files[dim]=set()
        output_files[dim].add(btch_fn)
        lastjobid+=1
    return joblist,lastjobid

def run_job(job_name,highest_dimension,calcs_per_batch,peroidic_unit_size,write_freq=1,useMPI=True):
    global output_files
    job_options={
        "calcs_per_batch":calcs_per_batch,
        "peroidic_unit_size":peroidic_unit_size,
        "write_freq":write_freq,
        "job_name":job_name
        }
    lastrunid={highest_dimension:0,highest_dimension-1:0}
    is_total_conv=False
    total_conv_checker={highest_dimension:convcrit.Overall_Diffusion_Conv(),highest_dimension-1:convcrit.Overall_Diffusion_Conv()}
    loop_count=0
    conv_by_dim={highest_dimension:False,highest_dimension-1:False}
    while(not is_total_conv):
        mpi_job_list=[]
        for i in range(2):
            if(not conv_by_dim[highest_dimension-i]):
                jlist,lastid=run_dim_batch(highest_dimension-i,job_options,batchnumber=loop_count,lastjobid=lastrunid[highest_dimension-i])
                mpi_job_list+=jlist
                lastrunid[highest_dimension-i]=lastid
        result=[]
        if(useMPI):
            from mpi4py.futures import MPIPoolExecutor
            with MPIPoolExecutor() as pool:
                result = pool.map(single_diffusion_calc, mpi_job_list)
        else:
            for jp in mpi_job_list:
                result.append(single_diffusion_calc(jp))
        temp_cov=True
        rel_std_list={}
        for i in range(2):
            conv,rel_stds=total_conv_checker[highest_dimension-i].check_conv(output_files[highest_dimension-i])
            rel_std_list[highest_dimension-i]=rel_stds
            conv_by_dim[highest_dimension-i]=conv
            print(rel_std_list)
            temp_cov=temp_cov and conv_by_dim[highest_dimension-i]
        is_total_conv=temp_cov
        reslog([is_total_conv,conv_by_dim,loop_count,rel_std_list])
        loop_count+=1
                
if(__name__=="__main__"):
	run_job("test0",5,10,3,useMPI=False)
