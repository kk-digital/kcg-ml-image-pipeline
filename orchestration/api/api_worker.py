from fastapi import Request, APIRouter, HTTPException, Query, Body
from utility.minio import cmd
from .api_utils import PrettyJSONResponse
import json
import paramiko
import csv

router = APIRouter()

@router.post("/worker-stats", response_class=PrettyJSONResponse)
def get_worker_stats(ssh_key_path: str = Query(...), 
                     server_address: str = Query("123.176.98.90")):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    gpu_stats_all = []
    port_numbers = [40029, 40030, 40132]  

    try:
        for port_number in port_numbers:
            try:
                ssh.connect(server_address, port=port_number, username='root', key_filename=ssh_key_path)

                command = '''
                python -c "import json; import socket; import GPUtil; print(json.dumps([{\'temperature\': gpu.temperature, \'load\': gpu.load, \'total_memory\': gpu.memoryTotal, \'used_memory\': gpu.memoryUsed, \'worker_name\': socket.gethostname()} for gpu in GPUtil.getGPUs()]))"
                '''
                stdin, stdout, stderr = ssh.exec_command(command)

                stderr_output = stderr.read().decode('utf-8')
                if stderr_output:
                    print(f"Error on server {server_address}:{port_number}:", stderr_output)
                    continue

                gpu_stats = json.loads(stdout.read().decode('utf-8'))
                gpu_stats_all.extend(gpu_stats)

            finally:
                ssh.close()

        return gpu_stats_all

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))