FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    slurm-wlm munge libmunge2 libmunge-dev \
    curl ca-certificates vim tini procps git sudo \
    fuse3 s3fs xz-utils tar \
    && rm -rf /var/lib/apt/lists/*

# Create system users
RUN useradd -r -m -s /bin/bash slurm && \
    useradd -r -m -s /bin/bash munge && \
    useradd -m -s /bin/bash runner

# Create necessary directories with proper ownership
RUN mkdir -p /etc/slurm /var/spool/slurm /var/log/slurm /etc/munge /var/lib/munge && \
    chown -R slurm:slurm /var/spool/slurm /var/log/slurm && \
    chown -R munge:munge /etc/munge /var/lib/munge

# Configure fuse
RUN echo "user_allow_other" >> /etc/fuse.conf

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt /app/
COPY slurm_api/ /app/slurm_api/
COPY conf/ /app/conf/
COPY scripts/ /app/scripts/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Make scripts executable
RUN chmod +x /app/scripts/entrypoint.sh

# Set environment defaults
ENV JR_AUTH_TOKEN=change-me
ENV JR_ROOT=/opt/jobrunner
ENV JR_NUM_GPUS=1
ENV JR_NUM_CPUS=8
ENV JR_REAL_MEM_MB=65536
ENV JR_GPU_TYPE=nvidia
ENV JR_MAX_TIME=7-00:00:00
ENV JR_USE_CGROUPS=0
ENV JR_B2_ENABLE=0
ENV JR_B2_BUCKET=
ENV JR_B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com
ENV JR_B2_KEY_ID=
ENV JR_B2_APP_KEY=
ENV JR_B2_MOUNT=/mnt/b2

# Expose API port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/scripts/entrypoint.sh"]
