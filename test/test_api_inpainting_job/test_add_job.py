import requests
import json



# Define the endpoint URL
url = "http://127.0.0.1:8000/queue/inpainting-generation/add-job"

# Prepare the files to be uploaded
mask_image = ("mask_image", open('test_inpainting_init_img_001.jpg', 'rb'))
init_image = ("init_image", open('test_inpainting_init_img_002.jpg', 'rb'))

# Define the task data as a dictionary
task_data = {
    "task_type": "inpainting-generation",
    "uuid": "1",
    "model_name": "1",
    "model_file_name": "1",
    "model_file_path": "1",
    "model_hash": "1",
    "task_creation_time": "1",
    "task_start_time": "1",
    "task_completion_time": "1",
    "task_error_str": "1",
    "task_input_dict": {
        "model": 'environmental',
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
    "task_output_file_dict": {},
    "task_attributes_dict": {},
    "prompt_generation_data": {},
}

# Create the payload data with the task field included as a dictionary
payload = {
    'task': json.dumps(task_data)
}

# Send the POST request with files and headers
response = requests.post(url, files={'mask_image': mask_image, 'input_image': init_image}, data=payload)

# Handle the response
if response.status_code == 200:
    print("Job added successfully!")
    print(response.json())
else:
    print("Error adding job:")
    print(response.text)

