from minio import Minio
from minio.error import InvalidResponseError
import requests
from utility.minio import cmd

# MinIO server information
MINIO_ADDRESS = "123.176.98.90:9000"
access_key = "GXvqLWtthELCaROPITOG"
secret_key = "DmlKgey5u0DnMHP30Vg7rkLT0NNbNIGaM8IwPckD"
secure = False 

# Initialize the MinIO client
client = Minio(MINIO_ADDRESS, access_key, secret_key, secure=secure)

#Check server status
try:
    response = requests.get("http://" + MINIO_ADDRESS + "/minio/health/live", timeout=5)
    if response.status_code == 200:
        print("MinIO server is accessible.")
    else:
        print("MinIO server is not accessible. Status code:", response.status_code)
except requests.RequestException as e:
    print("Failed to connect to MinIO server:", str(e))




