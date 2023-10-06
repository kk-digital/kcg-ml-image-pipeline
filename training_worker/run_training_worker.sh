#!/bin/bash

# run and pass the args given
python3 ./training_worker/training_worker.py --minio-access-key $1 --minio-secret-key $2 --worker-type $3