#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/training/add -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "ab_ranking_linear_task",
        "dataset_name": "icons",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_output_file_dict": {}
    }'