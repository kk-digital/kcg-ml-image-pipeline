import random
import os
import sys
base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from utility.minio import cmd
from training_worker.classifiers.models.elm_regression import ELMRegression
from minio import Minio
import torch
import msgpack
import csv


# Define MinIO access
MINIO_ADDRESS = '192.168.3.5:9000'
ACCESS_KEY = 'v048BpXpWrsVIHUfdAix'
SECRET_KEY = '4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu'
MODEL_DATASET = 'environmental'  
MODEL_TYPE = 'clip'
SCORING_MODEL = 'elm'
not_include = 'kandinsky'

def save_scores_to_csv(results, tag_dir, csv_filename):
    # Ensure the tag directory exists
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)
    # Path for the CSV file
    csv_path = os.path.join(tag_dir, csv_filename)
    # Create or overwrite the CSV file
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['image_path', 'actual_score', 'percentile', 'tag', 'model'])
        # Write the data rows
        for result in results:
            writer.writerow([result['image_path'], result['actual_score'], result['percentile'], result['tag'], result['model']])
    print(f"Saved classification results to {csv_path}.")

def save_image(local_dir, image_name, image_data):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)  # Create the directory if it does not exist
    file_path = os.path.join(local_dir, image_name)
    with open(file_path, 'wb') as file:
        file.write(image_data)
    print(f"Image saved to {file_path}.")

def get_clip_vectors(minio_client, base_path, num_samples=900):
    objects_list = list(minio_client.list_objects(bucket_name='datasets', prefix=base_path, recursive=False))
    clip_objects = [obj for obj in objects_list if obj.object_name.endswith('_clip.msgpack')]
    selected_clip_objects = random.sample(clip_objects, min(len(clip_objects), num_samples))

    clip_vectors_and_paths = []
    for clip_obj in selected_clip_objects:
        obj_data = minio_client.get_object('datasets', clip_obj.object_name)
        obj_content = obj_data.read()
        unpacked_data = msgpack.unpackb(obj_content, raw=False)
        vector = unpacked_data['clip-feature-vector'][0]
        vector_tensor = torch.tensor(vector).to(device)
        clip_vectors_and_paths.append((vector_tensor, clip_obj.object_name))  # Store both tensor and path

    return clip_vectors_and_paths

def calculate_percentiles(scores):
    sorted_scores = sorted(scores)
    percentile_bins = {p: [] for p in [0.1 * i for i in range(1, 11)]} 
    for score in scores:
        percentile = sum(s < score for s in sorted_scores) / len(sorted_scores)
        for bin in sorted(percentile_bins.keys(), reverse=True):
            if percentile >= bin:
                percentile_bins[bin].append(score)
                break
    return percentile_bins

def get_unique_tag_names(minio_client, model_dataset):
    prefix = f"{model_dataset}/models/classifiers/"
    objects = minio_client.list_objects(bucket_name='datasets', prefix=prefix, recursive=False)
    tag_names = set()  # Use a set to avoid duplicates
    for obj in objects:
        parts = obj.object_name.split('/')
        if len(parts) > 3:  # Ensures that the path is deep enough to include a tag_name
            tag_name = parts[3]  # Assumes tag_name is the fourth element in the path
            tag_names.add(tag_name)
    return list(tag_names)

# Initialize your ELMRegression model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print a message indicating the device being used
if torch.cuda.is_available():
    print("CUDA (GPU) is available. Using GPU for computations.")
else:
    print("CUDA (GPU) is not available. Using CPU for computations.")
    
# Initialize MinIO client
minio_client = cmd.connect_to_minio_client(MINIO_ADDRESS, access_key=ACCESS_KEY, secret_key=SECRET_KEY)



elm_model = ELMRegression(device=device)

tag_name_list = get_unique_tag_names(minio_client, MODEL_DATASET)
base_path = f"{MODEL_DATASET}/0356/"
clip_vectors_and_paths = get_clip_vectors(minio_client, base_path)

for tag_name in tag_name_list:
    # Load the model for the specified tag
    loaded_model, model_file_name = elm_model.load_model(minio_client, MODEL_DATASET, tag_name, MODEL_TYPE, SCORING_MODEL, not_include, device=device)

    if loaded_model:
        print(f"Model for tag '{tag_name}' loaded successfully.")
        
        scores = []
        path_to_score = {}
        

        for vector, path in clip_vectors_and_paths:
            try:
                
                vector = vector.to(device)
                
                vector = vector.unsqueeze(0)
                
                classification_score = loaded_model.classify(vector).item()  # Get the actual float value of the score
                
                scores.append(classification_score)
                image_path = path.replace('_clip.msgpack', '.jpg')  # Convert clip vector path to image path
                path_to_score[image_path] = classification_score
            except RuntimeError as e:
                print(f"Skipping vector due to error: {e}")
                continue
            
        percentile_bins = calculate_percentiles(scores)
        
        # Create a base directory for the current tag model
        tag_base_dir = os.path.join(base_directory, tag_name)

        csv_results = []

        for bin, bin_scores in percentile_bins.items():
            bin_dir = os.path.join(tag_base_dir, f"{bin:.1f}")
            for score in bin_scores:
                image_path = [path for path, s in path_to_score.items() if s == score][0]
                csv_results.append({
                    'image_path': image_path,
                    'actual_score': score,
                    'percentile': bin,
                    'tag': tag_name,
                    'model': model_file_name
                })
                image_data = cmd.get_file_from_minio(minio_client, 'datasets', image_path)
                if image_data:
                    image_content = image_data.read()
                    save_image(bin_dir, os.path.basename(image_path), image_content)
                else:
                    print(f"Failed to fetch image for score {score}.")
        
        # Save the results to a CSV file in the tag directory
        csv_filename = f"{tag_name}_classification_results.csv"
        save_scores_to_csv(csv_results, tag_base_dir, csv_filename)
    else:
        print(f"Failed to load the model for tag: {tag_name}.")