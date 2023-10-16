#!/bin/bash

# pass the first arg given as device
export CUDA_VISIBLE_DEVICES=0,1,2
python3 ./worker/worker.py --device $1 --minio-access-key $2 --minio-secret-key $3 --worker-type $4