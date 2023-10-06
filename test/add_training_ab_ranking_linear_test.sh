#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/training/add -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "ab_ranking_linear_task",
        "dataset_name": "icons",
        "epochs": 10,
        "learning_rate": 0.001,
        "buffer_size": 20000,
        "train_percent": 0.9,
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_output_file_dict": {}
    }'