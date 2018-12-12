from path import Path
import sys


try:
    batchsize = int(sys.argv[1])
except IndexError:
    batchsize = 1000
nclusters = len(Path("../data/clusters/").glob("*.xyz"))
nbatches = nclusters // batchsize + 1
nbatches = int(nbatches)
print(f"{nclusters} ~= {batchsize} x {nbatches}")

batch_sizes = [1000 for _ in range(nbatches)]
batch_sizes[-1] = nclusters % batchsize


for i in range(nbatches):
    bsize = batch_sizes[i]
    content = \
f"""
universe = vanilla
requirements = (OpSys == "LINUX") && (OpSysMajorVer == 6)
plus = $(Process) + {batchsize*i}
NewProcess = $INT(plus)
executable = condor_run.sh
arguments = $(NewProcess)
output = condor_outputs/$(Cluster)_$(NewProcess).out
error = condor_errors/$(Cluster)_$(NewProcess).err
log = $(Cluster).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = ../data/clusters.tar.gz, condor_run.sh
transfer_output_files = ppm3d_correct/analysis/data/results, ppm3d_correct/analysis/data/errors
request_cpus = 1
request_memory = 2GB
request_disk = 2GB
queue {bsize}
"""

    open(f"submit{i}.sub", "w").write(content)
