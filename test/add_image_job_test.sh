#!/bin/bash

curl -X 'POST' http://192.168.3.1:8111/queue/image-generation/add -H 'Content-Type: application/json' -d '{
        "uuid": "1234567",
        "task_type": "image_generation_task_test",
        "model_name": "v1-5-pruned-emaonly",
        "model_file_name": "v1-5-pruned-emaonly",
        "model_file_path": "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            "positive_prompt": "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, russian futurism",
            "negative_prompt": "low resolution, mediocre style, normal resolution",
            "cfg_strength": 12,
            "seed": "",
            "dataset": "test-generations",
            "file_path": "[auto]",
            "num_images": 1,
            "image_width": 512,
            "image_height": 512,
            "sampler": "ddim",
            "sampler_steps": 20,
            "prompt_scoring_model": "test-scoring-model",
            "prompt_score": 1000,
            "prompt_generation_policy": "N/A",
            "top_k": 0
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'