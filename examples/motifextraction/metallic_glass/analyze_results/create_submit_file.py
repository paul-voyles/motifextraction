from path import Path
from natsort import natsorted

files = natsorted(Path('../data/averaged/').glob('*.xyz'))

content = \
"""
universe = vanilla
requirements = (OpSys == "LINUX") && (OpSysMajorVer == 6)
executable = condor_run.sh
arguments = $(Process)
output = condor_outputs/$(Cluster)_$(Process).out
error = condor_errors/$(Cluster)_$(Process).err
log = $(Cluster).log
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = ../data/clusters.tar.gz, condor_run.sh, ../data/averaged
transfer_output_files = ppm3d_correct/analysis/data/motif_results, ppm3d_correct/analysis/data/motif_errors
request_cpus = 1
request_memory = 2GB
request_disk = 2GB
queue arguments from (
{}
)
""".format("\n".join(files))

open("submit.sub", "w").write(content)