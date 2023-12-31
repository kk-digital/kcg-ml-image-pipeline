#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/job/add -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "generate_inpainting_generation_task",
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
            "dataset_name": "character",
            "positive_prefix": "chibi, side scrolling",
            "init_img_path": "./test/test_inpainting/white_512x512.jpg",
            "mask_path": "./test/test_inpainting/character_mask.png"
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'





curl -X 'POST' http://192.168.3.1:8111/job/add -H 'Content-Type: application/json' -d '{
        "uuid": "",
        "task_type": "generate_inpainting_generation_task",
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
            "dataset_name": "icons",
            "positive_prefix": "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, futurism",
            "init_img_path": "./test/test_inpainting/white_512x512.jpg",
            "mask_path": "./test/test_inpainting/icon_mask.png"
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'