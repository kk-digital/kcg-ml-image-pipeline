#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/add-job -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "clip_calculation_task",
        "model_name": "",
        "model_file_name": "",
        "model_file_path": "",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            "input_file_path": "datasets/icons/0001/000008.jpg",
            "input_file_hash": "95980126540a2032d8e4133ffb6029ae4dfe2a2e9e9214649c6a45cab0128b60"
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'