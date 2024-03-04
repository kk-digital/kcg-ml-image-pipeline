import requests

# Define the URL of your FastAPI server
url = "http://127.0.0.1:8000/queue/inpainting-generation/add-job"  # Update with your server's URL

# Define the task data and image files to upload
task_data = {"task": {
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
            "dataset": "environmental",
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
}}
mask_image = ("test_inpainting_init_img_001.jpg", open("test_inpainting_init_img_001.jpg", "rb"))
input_image = ("test_inpainting_init_img_002.jpg", open("test_inpainting_init_img_002.jpg", "rb"))

# Make a POST request to the FastAPI endpoint
response = requests.post(url, data=task_data, files={"mask_image": mask_image, "input_image": input_image})

# Print the response from the server
print(response.json())