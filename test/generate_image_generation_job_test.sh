#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/job/add -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "generate_image_generation_task",
        "model_name": "v1-5-pruned-emaonly",
        "model_file_name": "v1-5-pruned-emaonly",
        "model_file_path": "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            "csv_dataset_path": "input/civitai_phrases_database_v6.csv",
            "prompt_count": 10,
            "dataset_name": "environmental",
            "positive_prefix": "environmental, pixel art, concept art, side scrolling, video game"
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'