import subprocess


lattice_params=[]
use_minmax=False
if(use_minmax):
	min_lattice=3
	max_lattice=11
	lattice_params=range(min_lattice,max_lattice+1)
else:
	lattice_params=[3,5,7,11,27,51,101,201,501,1001]
max_dim=2
cutoff=11
jobname="cgsv2"
template="tmplt.sh"

repl={
"{par}":"et3,et2024",
"{cores}":50,
"{maxdim}":max_dim,
"{cutoff}":cutoff,
"{jobname}":jobname
}

subname_tmplt=jobname+"_np-{latnode}_cutoff-{cutoff}_mdim-{maxdim}.sh"

f=open(template,"r")
tmplt=f.read()
f.close()


for i in lattice_params:
	repl["{latnode}"]=i
	subname=subname_tmplt
	jobfile=tmplt
	for k in repl:
		subname=subname.replace(k,str(repl[k]))
		jobfile=jobfile.replace(k,str(repl[k]))
	f=open(subname,"w+")
	f.write(jobfile)
	f.close()
	out=subprocess.check_output("sbatch "+subname,shell=True).decode("utf-8")
	print(out)
