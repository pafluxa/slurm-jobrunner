#!/usr/bin/env python3
import os
import socket

def main():
    # Read environment variables
    hostname = os.environ.get('JR_HOSTNAME', socket.gethostname())
    num_gpus = os.environ.get('JR_NUM_GPUS', '1')
    num_cpus = os.environ.get('JR_NUM_CPUS', '8')
    real_mem_mb = os.environ.get('JR_REAL_MEM_MB', '65536')
    max_time = os.environ.get('JR_MAX_TIME', '7-00:00:00')
    gpu_type = os.environ.get('JR_GPU_TYPE', 'nvidia')
    use_cgroups = os.environ.get('JR_USE_CGROUPS', '0')
    gpu_dev_prefix = os.environ.get('JR_GPU_DEV_PREFIX', '/dev/nvidia')
    
    # Determine cgroup settings
    if use_cgroups.lower() in {'1', 'true', 'yes'}:
        proctrack = 'proctrack/cgroup'
        cg_constrain_cores = 'yes'
        cg_constrain_devices = 'yes'
        cg_constrain_ram = 'yes'
        cg_constrain_swap = 'yes'
    else:
        proctrack = 'proctrack/linuxproc'
        cg_constrain_cores = 'no'
        cg_constrain_devices = 'no'
        cg_constrain_ram = 'no'
        cg_constrain_swap = 'no'
    
    # Generate slurm.conf
    with open('/app/conf/slurm.conf.tpl', 'r') as f:
        slurm_conf_template = f.read()
    
    slurm_conf = slurm_conf_template.replace('{{HOSTNAME}}', hostname)
    slurm_conf = slurm_conf.replace('{{NUM_GPUS}}', num_gpus)
    slurm_conf = slurm_conf.replace('{{NUM_CPUS}}', num_cpus)
    slurm_conf = slurm_conf.replace('{{REAL_MEM_MB}}', real_mem_mb)
    slurm_conf = slurm_conf.replace('{{MAX_TIME}}', max_time)
    slurm_conf = slurm_conf.replace('{{GPU_TYPE}}', gpu_type)
    slurm_conf = slurm_conf.replace('{{PROCTRACK}}', proctrack)
    
    with open('/etc/slurm/slurm.conf', 'w') as f:
        f.write(slurm_conf)
    
    # Generate gres.conf
    gres_lines = []
    for i in range(int(num_gpus)):
        gres_lines.append(f'Name=gpu Type={gpu_type} File={gpu_dev_prefix}{i}')
    
    with open('/app/conf/gres.conf.tpl', 'r') as f:
        gres_conf_template = f.read()
    
    gres_conf = gres_conf_template.replace('{{GRES_LINES}}', '\n'.join(gres_lines))
    
    with open('/etc/slurm/gres.conf', 'w') as f:
        f.write(gres_conf)
    
    # Generate cgroup.conf
    with open('/app/conf/cgroup.conf.tpl', 'r') as f:
        cgroup_conf_template = f.read()
    
    cgroup_conf = cgroup_conf_template.replace('{{CG_CONSTRAIN_CORES}}', cg_constrain_cores)
    cgroup_conf = cgroup_conf.replace('{{CG_CONSTRAIN_DEVICES}}', cg_constrain_devices)
    cgroup_conf = cgroup_conf.replace('{{CG_CONSTRAIN_RAM}}', cg_constrain_ram)
    cgroup_conf = cgroup_conf.replace('{{CG_CONSTRAIN_SWAP}}', cg_constrain_swap)
    
    with open('/etc/slurm/cgroup.conf', 'w') as f:
        f.write(cgroup_conf)
    
    # Copy cgroup allowed devices file
    with open('/app/conf/cgroup_allowed_devices_file.conf', 'r') as f:
        cgroup_allowed = f.read()
    
    with open('/etc/slurm/cgroup_allowed_devices_file.conf', 'w') as f:
        f.write(cgroup_allowed)
    
    print(f"Generated Slurm configuration for {hostname}")
    print(f"  GPUs: {num_gpus} x {gpu_type}")
    print(f"  CPUs: {num_cpus}")
    print(f"  Memory: {real_mem_mb} MB")
    print(f"  Max time: {max_time}")
    print(f"  Cgroups: {'enabled' if use_cgroups.lower() in {'1', 'true', 'yes'} else 'disabled'}")

if __name__ == '__main__':
    main()
