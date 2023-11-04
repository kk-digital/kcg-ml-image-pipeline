#!/bin/bash

curl -X 'POST' http://127.0.0.1:8000/queue/image-generation/add -H 'Content-Type: application/json' -d '{
        "uuid": "1234567",
        "task_type": "inpainting_generation_task",
        "model_name" : "v1-5-pruned-emaonly",
        "model_file_name": "v1-5-pruned-emaonly",
        "model_file_path": "input/model/sd/v1-5-pruned-emaonly/v1-5-pruned-emaonly.safetensors",
        "sd_model_hash": "N/A",
        "task_creation_time": "N/A",
        "task_start_time": "N/A",
        "task_completion_time": "N/A",
        "task_error_str": "",
        "task_input_dict": {
            "positive_prompt": "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, russian futurism",
            "negative_prompt" : "low resolution, mediocre style, normal resolution",
            "cfg_strength": 12,
            "seed": "",
            "dataset": "test-generations",
            "file_path": "[auto]",
            "image_width": 512,
            "image_height": 512,
            "sampler": "ddim",
            "sampler_steps": 20,
            "init_img": "./test/test_inpainting/white_512x512.jpg",
            "init_mask": "./test/test_inpainting/icon_mask.png",
            "mask_blur" : 0,
            "inpainting_fill_mode": 1,
            "styles": [],
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "image_cfg_scale": 1.5,
            "inpaint_full_res_padding": 32,
            "inpainting_mask_invert": 0
        },
        "task_input_file_dict": {},
        "task_output_file_dict": {}
    }'