import requests
import mimetypes
# Define the URL of your FastAPI server
url = "http://127.0.0.1:8000/queue/inpainting-generation/add-job"  # Update with your server's URL

# Define the task data and image files to upload
task_data = {
        "task_type": "your_task_type_here",
        "uuid": "your_generated_uuid_here",
        "model_name": "your_model_name_here",
        "model_file_name": "your_model_file_name_here",
        "task_input_dict": {
            "dataset": "environmental"
        }
    }
mask_image = ("test_inpainting_init_img_001.jpg", open("test_inpainting_init_img_001.jpg", "rb"))
input_image = ("test_inpainting_init_img_002.jpg", open("test_inpainting_init_img_002.jpg", "rb"))
content_type, _ = mimetypes.guess_type(mask_image[0])
headers = {"Content-Type": f"multipart/form-data; boundary={content_type}"}

# Make a POST request to the FastAPI endpoint
response = requests.post(url, json=task_data, files={"mask_image": mask_image, "input_image": input_image}, headers=headers)

# Print the response from the server
print(response.json())
