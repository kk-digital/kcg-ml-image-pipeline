#!/bin/bash

# pass the first arg given as device
python3 ./worker/worker.py --device $1 --minio-access-key $2 --minio-secret-key $3