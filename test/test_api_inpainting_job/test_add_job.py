import requests
import json

# Define the task data
task_data = {
    "task_type": "your_task_type_here",
    "uuid": "your_generated_uuid_here",
    "model_name": "your_model_name_here",
    "model_file_name": "your_model_file_name_here",
    "task_input_dict": {
        "dataset": "environmental"
    }
}

# Define the endpoint URL
url = "http://127.0.0.1:8000/queue/inpainting-generation/add-job"

# Prepare the files to be uploaded
mask_image = open('test_inpainting_init_img_001.jpg', 'rb')
init_image = open('test_inpainting_init_img_002.jpg', 'rb')

# Create the payload data
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

# Close the file handles
mask_image.close()
init_image.close()
