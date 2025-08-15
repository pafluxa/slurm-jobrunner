#!/bin/bash
set -e

# Export environment defaults from Dockerfile
export JR_AUTH_TOKEN=${JR_AUTH_TOKEN:-change-me}
export JR_ROOT=${JR_ROOT:-/opt/jobrunner}
export JR_NUM_GPUS=${JR_NUM_GPUS:-1}
export JR_NUM_CPUS=${JR_NUM_CPUS:-8}
export JR_REAL_MEM_MB=${JR_REAL_MEM_MB:-65536}
export JR_GPU_TYPE=${JR_GPU_TYPE:-nvidia}
export JR_MAX_TIME=${JR_MAX_TIME:-7-00:00:00}
export JR_USE_CGROUPS=${JR_USE_CGROUPS:-0}
export JR_B2_ENABLE=${JR_B2_ENABLE:-0}
export JR_B2_BUCKET=${JR_B2_BUCKET:-}
export JR_B2_ENDPOINT=${JR_B2_ENDPOINT:-https://s3.us-west-004.backblazeb2.com}
export JR_B2_KEY_ID=${JR_B2_KEY_ID:-}
export JR_B2_APP_KEY=${JR_B2_APP_KEY:-}
export JR_B2_MOUNT=${JR_B2_MOUNT:-/mnt/b2}
export PORT=${PORT:-8080}

# Create jobrunner directories
echo "Creating jobrunner directories..."
mkdir -p ${JR_ROOT}/{datasets,results,logs}
chown -R runner:runner ${JR_ROOT}

# Initialize Munge
echo "Initializing Munge..."
if [ ! -f /etc/munge/munge.key ]; then
    /usr/sbin/create-munge-key -f
fi
chmod 400 /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
service munge start

# Generate Slurm configuration
echo "Generating Slurm configuration..."
python3 /app/scripts/gen_slurm_conf.py

# Set ownership for Slurm directories
chown -R slurm:slurm /var/spool/slurm /var/log/slurm

# Mount B2 if enabled
B2_MOUNTED=false
if [ "${JR_B2_ENABLE}" = "1" ] && [ -n "${JR_B2_KEY_ID}" ] && [ -n "${JR_B2_APP_KEY}" ] && [ -n "${JR_B2_BUCKET}" ]; then
    echo "Mounting B2 bucket ${JR_B2_BUCKET}..."
    
    # Create password file for s3fs
    echo "${JR_B2_KEY_ID}:${JR_B2_APP_KEY}" > /etc/passwd-s3fs
    chmod 600 /etc/passwd-s3fs
    
    # Create mount point
    mkdir -p ${JR_B2_MOUNT}
    
    # Mount B2 bucket
    if s3fs ${JR_B2_BUCKET} ${JR_B2_MOUNT} \
        -o url=${JR_B2_ENDPOINT} \
        -o use_path_request_style \
        -o allow_other \
        -o umask=0002 \
        -o mp_umask=0002 \
        -o passwd_file=/etc/passwd-s3fs; then
        echo "B2 bucket mounted successfully"
        B2_MOUNTED=true
    else
        echo "Failed to mount B2 bucket, continuing without it"
    fi
fi

# Setup trap to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    if [ -n "${SLURMCTLD_PID}" ]; then
        kill ${SLURMCTLD_PID} 2>/dev/null || true
    fi
    if [ -n "${SLURMD_PID}" ]; then
        kill ${SLURMD_PID} 2>/dev/null || true
    fi
    if [ "${B2_MOUNTED}" = "true" ]; then
        echo "Unmounting B2..."
        fusermount -u ${JR_B2_MOUNT} 2>/dev/null || true
    fi
    exit 0
}
trap cleanup EXIT INT TERM

# Start Slurm services in foreground
echo "Starting slurmctld..."
slurmctld -Dvvv -f /etc/slurm/slurm.conf &
SLURMCTLD_PID=$!

echo "Starting slurmd..."
slurmd -Dvvv -f /etc/slurm/slurm.conf &
SLURMD_PID=$!

# Wait for Slurm to initialize
sleep 2

# Check Slurm status
echo "Checking Slurm status..."
sinfo || true

# Start FastAPI application
echo "Starting FastAPI application on port ${PORT}..."
cd /app
exec uvicorn slurm_api.slurm_api:app --host 0.0.0.0 --port ${PORT}
