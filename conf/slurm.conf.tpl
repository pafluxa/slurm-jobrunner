ClusterName=vast1
SlurmctldHost={{HOSTNAME}}
SlurmUser=slurm
SlurmctldPort=6817
SlurmdPort=6818
StateSaveLocation=/var/spool/slurm
SlurmdSpoolDir=/var/spool/slurm
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid

SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory,CR_Socket,CR_Native
SchedulerType=sched/backfill
ReturnToService=2
ProctrackType={{PROCTRACK}}

AccountingStorageType=accounting_storage/filetxt
AccountingStorageLoc=/var/log/slurm
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/jobcomp.log

NodeName={{HOSTNAME}} Gres=gpu:{{GPU_TYPE}}:{{NUM_GPUS}} RealMemory={{REAL_MEM_MB}} CPUs={{NUM_CPUS}} State=UNKNOWN
PartitionName=main Nodes={{HOSTNAME}} Default=YES MaxTime={{MAX_TIME}} State=UP
