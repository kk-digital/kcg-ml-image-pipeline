#!/bin/bash

# pass the first arg given as device
python3 ./prompt_job_generator/prompt_job_generator.py --device $1 --minio_access_key $2 --minio_secret_key $3 --csv_dataset_path $4