# Slurm JobRunner

A production-ready, single-node Slurm service containerized for Vast.ai (or any NVIDIA GPU host). Exposes a FastAPI interface for dataset management, job submission, and output retrieval.

## Features

- **Self-contained Slurm cluster** in a single Docker container
- **FastAPI interface** for programmatic job management  
- **Content-addressed dataset storage** with automatic archive extraction
- **Non-root job execution** for security
- **Backblaze B2/S3 integration** via s3fs mounting
- **SQLite-based metadata indexing** (no external database required)
- **Manifest-based output tracking** with heuristic fallback
- **GPU support** with automatic NVIDIA device detection

## Quick Start

### Build

```bash
docker build -t yourname/slurm-jobrunner:latest .
```

### Run on Vast.ai

```bash
docker run --gpus all --privileged \
  -e JR_AUTH_TOKEN=super-secret \
  -e JR_NUM_GPUS=$(nvidia-smi -L | wc -l) \
  -e JR_NUM_CPUS=$(nproc) \
  -e JR_REAL_MEM_MB=256000 \
  -e JR_B2_ENABLE=0 \
  -p 8080:8080 \
  yourname/slurm-jobrunner:latest
```

### With Persistence

To persist datasets and results across container restarts:

```bash
docker run --gpus all --privileged \
  -v /host/data:/opt/jobrunner \
  -e JR_AUTH_TOKEN=super-secret \
  -e JR_NUM_GPUS=$(nvidia-smi -L | wc -l) \
  -e JR_NUM_CPUS=$(nproc) \
  -e JR_REAL_MEM_MB=256000 \
  -p 8080:8080 \
  yourname/slurm-jobrunner:latest
```

### With B2 Storage

To enable Backblaze B2/S3 mounting:

```bash
docker run --gpus all --privileged \
  -e JR_AUTH_TOKEN=super-secret \
  -e JR_B2_ENABLE=1 \
  -e JR_B2_BUCKET=my-bucket \
  -e JR_B2_KEY_ID=your-key-id \
  -e JR_B2_APP_KEY=your-app-key \
  -e JR_B2_ENDPOINT=https://s3.us-west-004.backblazeb2.com \
  -e JR_NUM_GPUS=$(nvidia-smi -L | wc -l) \
  -e JR_NUM_CPUS=$(nproc) \
  -p 8080:8080 \
  yourname/slurm-jobrunner:latest
```

**Note:** B2/S3 mounting requires:
- Container run with `--privileged` flag
- Access to `/dev/fuse` device
- Valid B2/S3 credentials

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JR_AUTH_TOKEN` | `change-me` | Bearer token for API authentication |
| `JR_ROOT` | `/opt/jobrunner` | Root directory for datasets/results/logs |
| `JR_NUM_GPUS` | `1` | Number of GPUs available |
| `JR_NUM_CPUS` | `8` | Number of CPUs available |
| `JR_REAL_MEM_MB` | `65536` | Memory in MB |
| `JR_GPU_TYPE` | `nvidia` | GPU type identifier |
| `JR_MAX_TIME` | `7-00:00:00` | Maximum job time (7 days) |
| `JR_USE_CGROUPS` | `0` | Enable cgroups for resource control |
| `JR_B2_ENABLE` | `0` | Enable B2/S3 mounting |
| `JR_B2_BUCKET` | ` ` | B2/S3 bucket name |
| `JR_B2_ENDPOINT` | `https://s3.us-west-004.backblazeb2.com` | B2/S3 endpoint URL |
| `JR_B2_KEY_ID` | ` ` | B2/S3 access key ID |
| `JR_B2_APP_KEY` | ` ` | B2/S3 secret key |
| `JR_B2_MOUNT` | `/mnt/b2` | B2/S3 mount point |

## API Usage

All endpoints (except `/health`) require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer super-secret" http://localhost:8080/...
```

### Health Check

```bash
curl http://localhost:8080/health
```

### Upload Dataset

Upload a file or archive (auto-extracted):

```bash
# Create test archive
echo "test data" > data.txt
tar -czf data.tar.gz data.txt
sha=$(sha256sum data.tar.gz | awk '{print $1}')

# Upload
curl -H "Authorization: Bearer super-secret" \
  -F "file=@data.tar.gz" \
  -F "sha256=$sha" \
  http://localhost:8080/datasets/upload
```

### Submit Job

Submit a job using an uploaded dataset:

```bash
curl -H "Authorization: Bearer super-secret" \
  -F "dataset_sha=$sha" \
  -F "command_key=pytorch_example" \
  -F "gpus=1" \
  -F "cpus=2" \
  -F "mem_gb=8" \
  -F "time_limit=00:10:00" \
  http://localhost:8080/jobs
```

Available command keys:
- `echo_params`: Simple test that echoes parameters
- `pytorch_example`: PyTorch GPU test with weight saving
- `pytorch_infer`: Run custom inference script

### Check Job Status

```bash
job_id=1  # From submit response
curl -H "Authorization: Bearer super-secret" \
  http://localhost:8080/jobs/$job_id
```

### Get Job Logs

```bash
curl -H "Authorization: Bearer super-secret" \
  http://localhost:8080/jobs/$job_id/log
```

### Get Job Results

```bash
curl -H "Authorization: Bearer super-secret" \
  http://localhost:8080/jobs/$job_id/result
```

### List Outputs

```bash
# By job
curl -H "Authorization: Bearer super-secret" \
  http://localhost:8080/outputs/by_job/$job_id

# By experiment
curl -H "Authorization: Bearer super-secret" \
  "http://localhost:8080/outputs?experiment_id=job-1"
```

### Download Outputs

```bash
# By experiment/output ID
curl -L -H "Authorization: Bearer super-secret" \
  "http://localhost:8080/outputs/download?experiment_id=job-1&output_id=weights" \
  -o model.pth

# By job/path
curl -L -H "Authorization: Bearer super-secret" \
  "http://localhost:8080/outputs/download_by_job?job_id=1&rel_path=weights/model.pth" \
  -o model.pth
```

## Acceptance Tests

### 1. Health & Auth

```bash
# Health check (no auth required)
curl http://localhost:8080/health

# Auth required for other endpoints
curl -i http://localhost:8080/datasets/exists/abc  # Returns 401
```

### 2. Dataset Upload with Auto-extraction

```bash
# Create test archive
mkdir -p data
echo "test content" > data/file.txt
tar -cJf data.tar.xz data/

# Upload and verify
TOK=super-secret
BASE=http://localhost:8080
sha=$(sha256sum data.tar.xz | awk '{print $1}')

curl -H "Authorization: Bearer $TOK" \
  -F "file=@data.tar.xz" \
  -F "sha256=$sha" \
  $BASE/datasets/upload

curl -H "Authorization: Bearer $TOK" \
  $BASE/datasets/exists/$sha | grep '"exists": true'
```

### 3. Job Submission (Non-root)

```bash
# Submit job
resp=$(curl -s -H "Authorization: Bearer $TOK" \
  -F dataset_sha=$sha \
  -F command_key=pytorch_example \
  -F gpus=1 -F cpus=2 -F mem_gb=8 \
  -F time_limit=00:10:00 \
  $BASE/jobs)

jid=$(echo $resp | sed -n 's/.*"job_id":[ ]*\([0-9]*\).*/\1/p')

# Poll until complete
while true; do
  status=$(curl -s -H "Authorization: Bearer $TOK" $BASE/jobs/$jid | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
  echo "Status: $status"
  [[ "$status" == "COMPLETED" ]] && break
  sleep 2
done

# Check logs
curl -H "Authorization: Bearer $TOK" $BASE/jobs/$jid/log
```

### 4. Results & Indexing

```bash
# Get result (triggers auto-indexing)
curl -H "Authorization: Bearer $TOK" $BASE/jobs/$jid/result

# List outputs
curl -H "Authorization: Bearer $TOK" $BASE/outputs/by_job/$jid

# Download weights
exp=$(curl -s -H "Authorization: Bearer $TOK" $BASE/outputs/by_job/$jid | jq -r '.outputs[0].experiment_id')
curl -L -H "Authorization: Bearer $TOK" \
  "$BASE/outputs/download?experiment_id=$exp&output_id=weights" \
  -o model.pth
```

## Architecture

### Directory Structure

```
/opt/jobrunner/
├── datasets/      # Content-addressed dataset storage
├── results/       # Job output directories
├── logs/          # Slurm job logs
└── jr_slurm.db    # SQLite metadata database
```

### Security

- All jobs run as non-root user `runner`
- Bearer token authentication for API access
- Path traversal protection for file operations
- Safe archive extraction with validation

### Dataset Management

- Content-addressed storage using SHA256
- Automatic archive extraction (tar, tar.xz, tar.lzma)
- Import from B2/S3 mounted paths
- Deduplication via hash-based storage

### Job Execution

- Jobs submitted to Slurm via `sbatch`
- Non-root execution as `runner` user
- Resource allocation (GPUs, CPUs, memory, time)
- Three built-in command templates:
  - `echo_params`: Parameter echo test
  - `pytorch_example`: GPU test with weight saving
  - `pytorch_infer`: Custom script execution

### Output Management

- Manifest-based output tracking (`manifest.json`)
- Heuristic indexing for non-manifest jobs
- Weight file detection (`.pt`, `.pth`, `.bin`, `.ckpt`, `.safetensors`, `.onnx`)
- SQLite indexing for fast queries

## Troubleshooting

### Slurm Not Starting

Check Slurm logs:
```bash
docker exec <container> cat /var/log/slurm/slurmctld.log
docker exec <container> cat /var/log/slurm/slurmd.log
```

### B2 Mount Failed

- Ensure container run with `--privileged`
- Verify B2 credentials are correct
- Check `/dev/fuse` is accessible
- Review container logs for mount errors

### Jobs Stuck in PENDING

- Check available resources match job requirements
- Verify Slurm node is in UP state: `docker exec <container> sinfo`
- Review slurmctld logs for scheduling issues

### GPU Not Available

- Ensure container run with `--gpus all`
- Verify NVIDIA drivers installed on host
- Check GPU visibility: `docker exec <container> nvidia-smi`

## Development

### Running Tests

```bash
# Build image
docker build -t slurm-jobrunner:test .

# Run container
docker run -d --name jr-test --gpus all --privileged \
  -e JR_AUTH_TOKEN=test-token \
  -p 8080:8080 \
  slurm-jobrunner:test

# Run acceptance tests
./run_tests.sh

# Cleanup
docker stop jr-test && docker rm jr-test
```

### Adding Custom Commands

Edit `sbatch_command_template()` in
