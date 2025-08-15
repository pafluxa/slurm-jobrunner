#!/usr/bin/env python3
import os
import json
import sqlite3
import hashlib
import tarfile
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

# Configuration from environment
JR_AUTH_TOKEN = os.environ.get('JR_AUTH_TOKEN', 'change-me')
JR_ROOT = Path(os.environ.get('JR_ROOT', '/opt/jobrunner'))
JR_B2_MOUNT = Path(os.environ.get('JR_B2_MOUNT', '/mnt/b2'))
JR_SUBMIT_USER = os.environ.get('JR_SUBMIT_USER', 'runner')

# Paths
DATASETS = JR_ROOT / 'datasets'
RESULTS = JR_ROOT / 'results'
LOGS = JR_ROOT / 'logs'
DB_PATH = JR_ROOT / 'jr_slurm.db'

# Archive extensions
ARCHIVE_EXTS = {'.tar.xz', '.txz', '.tar.lzma', '.tlz', '.tar'}
WEIGHT_EXTS = {'.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.onnx'}

app = FastAPI(title="Slurm JobRunner API")

# Database initialization
def init_db():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            sha256 TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            slurm_id INTEGER PRIMARY KEY,
            dataset_sha TEXT NOT NULL,
            command_key TEXT NOT NULL,
            params_json TEXT NOT NULL,
            result_dir TEXT NOT NULL,
            log_path TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS outputs (
            experiment_id TEXT NOT NULL,
            output_id TEXT NOT NULL,
            path TEXT NOT NULL,
            job_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (experiment_id, output_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Security dependency
def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify authorization token."""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")
    
    token = authorization.replace('Bearer ', '')
    if token != JR_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    return True

# Helper functions
def run_cmd_as_runner(cmd: List[str]) -> str:
    """Execute command as runner user."""
    full_cmd = ['sudo', '-u', JR_SUBMIT_USER] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout

def safe_join(base: Path, rel: str) -> Path:
    """Safely join paths, preventing traversal."""
    base = base.resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path traversal attempt")
    return target

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def safe_extract_tar(tar_path: Path, extract_to: Path):
    """Safely extract tar archive, preventing path traversal."""
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            # Validate member path
            member_path = Path(extract_to) / member.name
            try:
                member_path = member_path.resolve()
                if not str(member_path).startswith(str(extract_to.resolve())):
                    raise ValueError(f"Path traversal attempt: {member.name}")
            except:
                raise ValueError(f"Invalid path: {member.name}")
        
        # Extract all members
        tar.extractall(extract_to)

def sbatch_command_template(command_key: str, dataset_dir: Path, params_json: str, job_id_hint: str) -> str:
    """Generate command snippet for sbatch script based on command key."""
    
    if command_key == 'echo_params':
        return f'''
python3 -c "
import json
import os
params = json.loads(os.environ.get('JR_PARAMS', '{{}}'))
result = {{'ok': True, 'params': params}}
result_path = os.path.join(os.environ['RESULT_DIR'], 'result.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Wrote result to {{result_path}}')
"
'''
    
    elif command_key == 'pytorch_example':
        return f'''
python3 -c "
import json
import os
import sys

# Try to import torch
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
except ImportError:
    cuda_available = False
    device_count = 0
    device_names = []
    print('PyTorch not installed, continuing with mock data')

result_dir = os.environ['RESULT_DIR']
params = json.loads(os.environ.get('JR_PARAMS', '{{}}'))

# Create weights directory
weights_dir = os.path.join(result_dir, 'weights')
os.makedirs(weights_dir, exist_ok=True)

# Save a small tensor as weights
if 'torch' in sys.modules:
    tensor = torch.randn(10, 10)
    torch.save(tensor, os.path.join(weights_dir, 'model.pth'))
else:
    # Create a dummy file
    with open(os.path.join(weights_dir, 'model.pth'), 'wb') as f:
        f.write(b'DUMMY_WEIGHTS')

# Write result.json
result = {{
    'ok': True,
    'cuda_available': cuda_available,
    'device_count': device_count,
    'device_names': device_names,
    'params': params
}}
with open(os.path.join(result_dir, 'result.json'), 'w') as f:
    json.dump(result, f, indent=2)

# Write manifest.json
slurm_job_id = os.environ.get('SLURM_JOB_ID', '{job_id_hint}')
manifest = {{
    'experimentId': f'job-{{slurm_job_id}}',
    'outputs': [
        {{'id': 'summary', 'path': 'result.json'}},
        {{'id': 'weights', 'path': 'weights/model.pth'}}
    ]
}}
with open(os.path.join(result_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print('PyTorch example completed successfully')
"
'''
    
    elif command_key == 'pytorch_infer':
        params = json.loads(params_json)
        script = params.get('script', '')
        args = params.get('args', [])
        args_str = ' '.join(args) if args else ''
        
        return f'''
# Run the inference script
python3 {script} --data "{dataset_dir}" --params '{params_json}' {args_str}

# Ensure result.json exists
python3 -c "
import json
import os

result_dir = os.environ['RESULT_DIR']
result_path = os.path.join(result_dir, 'result.json')
if not os.path.exists(result_path):
    with open(result_path, 'w') as f:
        json.dump({{'ok': True}}, f, indent=2)
    print('Created minimal result.json')
"

# Ensure manifest.json exists
python3 -c "
import json
import os

result_dir = os.environ['RESULT_DIR']
manifest_path = os.path.join(result_dir, 'manifest.json')
if not os.path.exists(manifest_path):
    params = json.loads(os.environ.get('JR_PARAMS', '{{}}'))
    slurm_job_id = os.environ.get('SLURM_JOB_ID', '{job_id_hint}')
    experiment_id = params.get('experimentId', f'job-{{slurm_job_id}}')
    manifest = {{
        'experimentId': experiment_id,
        'outputs': [{{'id': 'result', 'path': 'result.json'}}]
    }}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Created minimal manifest.json')
"
'''
    
    else:
        raise ValueError(f"Unknown command key: {command_key}")

def index_job_outputs(job_id: int) -> Dict[str, Any]:
    """Index outputs for a job, using manifest if available, otherwise heuristics."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get job info
    cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
    job = cur.fetchone()
    if not job:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    result_dir = Path(job['result_dir'])
    if not result_dir.exists():
        conn.close()
        return {'experiment_id': None, 'count': 0}
    
    # Check for manifest
    manifest_path = result_dir / 'manifest.json'
    
    if manifest_path.exists():
        # Use manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        experiment_id = manifest.get('experimentId', f'job-{job_id}')
        outputs = manifest.get('outputs', [])
    else:
        # Use heuristics
        params = json.loads(job['params_json'])
        experiment_id = params.get('experimentId', f'job-{job_id}')
        
        # Find all files in result_dir
        outputs = []
        seen_ids = set()
        
        for file_path in result_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(result_dir)
                
                # Determine output_id
                if file_path.suffix in WEIGHT_EXTS:
                    output_id = file_path.stem
                else:
                    output_id = file_path.stem
                
                # De-duplicate
                if output_id not in seen_ids:
                    outputs.append({
                        'id': output_id,
                        'path': str(rel_path)
                    })
                    seen_ids.add(output_id)
    
    # Clear existing outputs for this job
    cur.execute('DELETE FROM outputs WHERE job_id = ?', (job_id,))
    
    # Insert new outputs
    count = 0
    for output in outputs:
        cur.execute('''
            INSERT OR REPLACE INTO outputs (experiment_id, output_id, path, job_id)
            VALUES (?, ?, ?, ?)
        ''', (experiment_id, output['id'], output['path'], job_id))
        count += 1
    
    conn.commit()
    conn.close()
    
    return {'experiment_id': experiment_id, 'count': count}

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint, does not require authentication."""
    # Check if Slurm is running
    try:
        subprocess.run(['sinfo'], check=True, capture_output=True)
        slurm_ok = True
    except:
        slurm_ok = False
    
    # Check if B2 is mounted
    b2_mounted = (JR_B2_MOUNT / '.mount_ok').exists() if os.environ.get('JR_B2_ENABLE') == '1' else False
    
    return {"ok": True, "slurm": slurm_ok, "b2_mounted": b2_mounted}

# Dataset endpoints
@app.get("/datasets/exists/{sha256}")
async def dataset_exists(sha256: str, _: bool = Depends(verify_auth)):
    """Check if a dataset with the given SHA256 exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute('SELECT * FROM datasets WHERE sha256 = ?', (sha256,))
    dataset = cur.fetchone()
    
    conn.close()
    
    if dataset:
        return {"exists": True, "info": dict(dataset)}
    else:
        return {"exists": False}

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    sha256: Optional[str] = Form(None),
    _: bool = Depends(verify_auth)
):
    """Upload a dataset file, compute SHA256, and store it. Extract archives automatically."""
    # Create temp file to compute hash
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        hasher = hashlib.sha256()
        
        # Stream file while computing hash
        while chunk := await file.read(8192):
            temp_file.write(chunk)
            hasher.update(chunk)
    
    try:
        # Compute file hash
        computed_sha256 = hasher.hexdigest()
        
        # Verify SHA256 if provided
        if sha256 and sha256 != computed_sha256:
            os.unlink(temp_path)
            raise HTTPException(
                status_code=400,
                detail=f"SHA256 mismatch. Provided: {sha256}, computed: {computed_sha256}"
            )
        
        # Ensure destination directory exists
        dest_dir = DATASETS / computed_sha256
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        original_filename = file.filename or "uploaded_file"
        file_ext = Path(original_filename).suffix.lower()
        
        # Determine if this is an archive
        is_archive = False
        for ext in ARCHIVE_EXTS:
            if original_filename.lower().endswith(ext):
                is_archive = True
                break
        
        if is_archive:
            # Extract archive
            try:
                safe_extract_tar(temp_path, dest_dir)
                rel_path = f"{computed_sha256}/"
                # Delete the archive after extraction
                os.unlink(temp_path)
            except Exception as e:
                os.unlink(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to extract archive: {str(e)}"
                )
        else:
            # Move file to destination
            dest_path = dest_dir / original_filename
            shutil.move(temp_path, dest_path)
            rel_path = f"{computed_sha256}/{original_filename}"
        
        # Get file size
        if is_archive:
            # Sum up sizes of all extracted files
            total_bytes = sum(f.stat().st_size for f in dest_dir.rglob('*') if f.is_file())
        else:
            total_bytes = (dest_dir / original_filename).stat().st_size
        
        # Insert into database
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        cur.execute('''
            INSERT OR IGNORE INTO datasets (sha256, rel_path, bytes)
            VALUES (?, ?, ?)
        ''', (computed_sha256, rel_path, total_bytes))
        
        conn.commit()
        conn.close()
        
        return {
            "sha256": computed_sha256,
            "stored_path": rel_path,
            "bytes": total_bytes
        }
        
    except Exception as e:
        # Clean up temp file if still exists
        if temp_path.exists():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/datasets/import_path")
async def import_path(
    src_path: str = Form(...),
    _: bool = Depends(verify_auth)
):
    """Import a file from an absolute path or relative to JR_B2_MOUNT."""
    try:
        # Resolve source path
        if os.path.isabs(src_path):
            source_path = Path(src_path)
        else:
            source_path = JR_B2_MOUNT / src_path
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source file not found: {src_path}")
        
        # Create temp file to compute hash
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        # Copy file to temp location and compute hash
        hasher = hashlib.sha256()
        with open(source_path, 'rb') as src, open(temp_path, 'wb') as dest:
            while chunk := src.read(8192):
                dest.write(chunk)
                hasher.update(chunk)
        
        computed_sha256 = hasher.hexdigest()
        
        # Ensure destination directory exists
        dest_dir = DATASETS / computed_sha256
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        original_filename = source_path.name
        file_ext = source_path.suffix.lower()
        
        # Determine if this is an archive
        is_archive = False
        for ext in ARCHIVE_EXTS:
            if original_filename.lower().endswith(ext):
                is_archive = True
                break
        
        if is_archive:
            # Extract archive
            try:
                safe_extract_tar(temp_path, dest_dir)
                rel_path = f"{computed_sha256}/"
                # Delete the archive after extraction
                os.unlink(temp_path)
            except Exception as e:
                os.unlink(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to extract archive: {str(e)}"
                )
        else:
            # Move file to destination
            dest_path = dest_dir / original_filename
            shutil.move(temp_path, dest_path)
            rel_path = f"{computed_sha256}/{original_filename}"
        
        # Get file size
        if is_archive:
            # Sum up sizes of all extracted files
            total_bytes = sum(f.stat().st_size for f in dest_dir.rglob('*') if f.is_file())
        else:
            total_bytes = (dest_dir / original_filename).stat().st_size
        
        # Insert into database
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        cur.execute('''
            INSERT OR IGNORE INTO datasets (sha256, rel_path, bytes)
            VALUES (?, ?, ?)
        ''', (computed_sha256, rel_path, total_bytes))
        
        conn.commit()
        conn.close()
        
        return {
            "sha256": computed_sha256,
            "stored_path": rel_path,
            "bytes": total_bytes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

# Job endpoints
@app.post("/jobs")
async def submit_job(
    dataset_sha: str = Form(...),
    command_key: str = Form(...),
    params_json: str = Form("{}"),
    gpus: int = Form(1),
    cpus: int = Form(4),
    mem_gb: int = Form(16),
    time_limit: str = Form("04:00:00"),
    _: bool = Depends(verify_auth)
):
    """Submit a Slurm job."""
    try:
        # Validate JSON
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in params_json")
        
        # Check if dataset exists
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM datasets WHERE sha256 = ?', (dataset_sha,))
        dataset = cur.fetchone()
        
        if not dataset:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Dataset with SHA256 {dataset_sha} not found")
        
        # Resolve dataset path
        rel_path = dataset['rel_path']
        if rel_path.endswith('/'):
            # It's a directory
            dataset_dir = DATASETS / rel_path
        else:
            # It's a file, use parent directory
            dataset_dir = (DATASETS / rel_path).parent
        
        # Create result directory
        epoch = int(time.time())
        pid = os.getpid()
        result_dir = RESULTS / f"job_{epoch}_{pid}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Set log path
        log_path = f"logs/slurm-%j.out"
        
        # Generate sbatch script
        job_id_hint = f"{epoch}_{pid}"
        sbatch_cmd = sbatch_command_template(command_key, dataset_dir, params_json, job_id_hint)
        
        # Create temporary sbatch script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as sbatch_file:
            sbatch_path = sbatch_file.name
            sbatch_file.write(f"""#!/bin/bash
#SBATCH -A none
#SBATCH -p main
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time_limit}
#SBATCH -o {log_path}

set -euo pipefail
export DATASET_DIR='{dataset_dir}'
export RESULT_DIR='{result_dir}'
export JR_PARAMS='{params_json}'
export PYTHONUNBUFFERED=1
echo "JOB_START $(date -Is)"
echo "DATASET_DIR=$DATASET_DIR"
echo "RESULT_DIR=$RESULT_DIR"
cd "$DATASET_DIR"
{sbatch_cmd}
echo "JOB_END $(date -Is)"
""")
        
        try:
            # Submit job
            output = run_cmd_as_runner(['sbatch', '--parsable', sbatch_path])
            # Parse job ID (remove any cluster suffix)
            slurm_id = int(output.strip().split(';')[0])
            
            # Insert job into database
            cur.execute('''
                INSERT OR REPLACE INTO jobs 
                (slurm_id, dataset_sha, command_key, params_json, result_dir, log_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (slurm_id, dataset_sha, command_key, params_json, str(result_dir), log_path.replace('%j', str(slurm_id))))
            
            conn.commit()
            
            return {"job_id": slurm_id, "status": "SUBMITTED"}
            
        finally:
            # Clean up temporary script
            os.unlink(sbatch_path)
            conn.close()
            
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: int, _: bool = Depends(verify_auth)):
    """Get job status from Slurm."""
    try:
        # Check if job exists in database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_dict = dict(job)
        
        # Try squeue first
        try:
            squeue_output = run_cmd_as_runner(['squeue', '-j', str(job_id), '-h', '-o', '%T|%M|%V|%R'])
            if squeue_output.strip():
                parts = squeue_output.strip().split('|')
                status = parts[0]
                done = status in ['COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT']
                return {"job_id": job_id, "status": status, "done": done, "meta": job_dict}
        except:
            pass
        
        # Try sacct if not in squeue
        try:
            sacct_output = run_cmd_as_runner(['sacct', '-j', str(job_id), '-n', '-o', 'State,Elapsed,Submit'])
            if sacct_output.strip():
                status = sacct_output.strip().split()[0]
                done = True  # If in sacct and not in squeue, job is done
                return {"job_id": job_id, "status": status, "done": done, "meta": job_dict}
        except:
            pass
        
        # Default response if neither squeue nor sacct provides status
        return {"job_id": job_id, "status": "UNKNOWN", "done": False, "meta": job_dict}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.get("/jobs/{job_id}/log")
async def get_job_log(job_id: int, _: bool = Depends(verify_auth)):
    """Stream the job's log file."""
    try:
        # Check if job exists in database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Construct log path
        log_path = job['log_path']
        if '%j' in log_path:
            log_path = log_path.replace('%j', str(job_id))
        
        log_file = JR_ROOT / log_path
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail=f"Log file not found for job {job_id}")
        
        return FileResponse(log_file)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job log: {str(e)}")

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, _: bool = Depends(verify_auth)):
    """Cancel a running job."""
    try:
        # Check if job exists in database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Cancel job
        run_cmd_as_runner(['scancel', str(job_id)])
        
        return {"job_id": job_id, "status": "CANCELLED_REQUESTED"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: int, _: bool = Depends(verify_auth)):
    """Get job result.json, triggering auto-indexing first."""
    try:
        # Auto-index outputs (best effort)
        try:
            index_job_outputs(job_id)
        except:
            # Ignore indexing errors
            pass
        
        # Check if job exists in database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Construct result.json path
        result_path = Path(job['result_dir']) / 'result.json'
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Result file not found for job {job_id}")
        
        return FileResponse(result_path)
    
# Output endpoints
@app.post("/outputs/index/{job_id}")
async def index_outputs(job_id: int, _: bool = Depends(verify_auth)):
    """Trigger (re)indexing of a job's outputs."""
    try:
        result = index_job_outputs(job_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.get("/outputs/by_job/{job_id}")
async def get_outputs_by_job(job_id: int, _: bool = Depends(verify_auth)):
    """Get all outputs for a specific job."""
    try:
        # Check if job exists
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Get outputs
        cur.execute('SELECT * FROM outputs WHERE job_id = ?', (job_id,))
        outputs = [dict(row) for row in cur.fetchall()]
        
        conn.close()
        
        return {
            "job_id": job_id,
            "result_dir": job['result_dir'],
            "outputs": outputs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get outputs: {str(e)}")

@app.get("/outputs")
async def list_outputs(
    experiment_id: Optional[str] = None,
    prefix: Optional[str] = None,
    _: bool = Depends(verify_auth)
):
    """List outputs, optionally filtered by experiment_id and/or path prefix."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        query = 'SELECT * FROM outputs'
        params = []
        
        where_clauses = []
        if experiment_id:
            where_clauses.append('experiment_id = ?')
            params.append(experiment_id)
        
        if prefix:
            where_clauses.append('path LIKE ?')
            params.append(f'{prefix}%')
        
        if where_clauses:
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        cur.execute(query, params)
        outputs = [dict(row) for row in cur.fetchall()]
        
        conn.close()
        
        return {"outputs": outputs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list outputs: {str(e)}")

@app.get("/outputs/download")
async def download_output(
    experiment_id: str,
    output_id: str,
    _: bool = Depends(verify_auth)
):
    """Download an output file by experiment_id and output_id."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Find the output
        cur.execute('''
            SELECT o.*, j.result_dir 
            FROM outputs o 
            JOIN jobs j ON o.job_id = j.slurm_id 
            WHERE o.experiment_id = ? AND o.output_id = ?
        ''', (experiment_id, output_id))
        
        output = cur.fetchone()
        
        if not output:
            conn.close()
            raise HTTPException(
                status_code=404, 
                detail=f"Output not found for experiment_id={experiment_id}, output_id={output_id}"
            )
        
        # Resolve file path
        result_dir = Path(output['result_dir'])
        file_path = safe_join(result_dir, output['path'])
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Output file not found: {output['path']}")
        
        conn.close()
        return FileResponse(file_path, filename=file_path.name)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download output: {str(e)}")

@app.get("/outputs/download_by_job")
async def download_output_by_job(
    job_id: int,
    rel_path: str,
    _: bool = Depends(verify_auth)
):
    """Download a file from a job's result directory by relative path."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Check if job exists
        cur.execute('SELECT * FROM jobs WHERE slurm_id = ?', (job_id,))
        job = cur.fetchone()
        
        if not job:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Resolve file path
        result_dir = Path(job['result_dir'])
        file_path = safe_join(result_dir, rel_path)
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")
        
        conn.close()
        return FileResponse(file_path, filename=file_path.name)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

# Main execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("slurm_api:app", host="0.0.0.0", port=port, reload=False)