import requests

# Define the URL of your FastAPI server
url = "http://127.0.0.1:8000/queue/inpainting-generation/add-job"  # Update with your server's URL

# Define the task data and image files to upload
task_data = {"dataset": "environmental"}
mask_image = ("test_inpainting_init_img_001.jpg", open("init_img.jpg", "rb"))
input_image = ("test_inpainting_init_img_002.jpg", open("init_mask.jpg", "rb"))

# Make a POST request to the FastAPI endpoint
response = requests.post(url, data=task_data, files={"mask_image": mask_image, "input_image": input_image})

# Print the response from the server
print(response.json())