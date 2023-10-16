#!/bin/bash

# pass the first arg given as device
python3 ./prompt_job_generator/prompt_job_generator.py --device $1 --minio-access-key $2 --minio-secret-key $3 --csv_dataset_path $4