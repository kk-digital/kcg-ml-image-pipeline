import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import io
import json
import os
import random
import sys
import time
import msgpack
import requests
import torch
from tqdm import tqdm
import faiss
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
from kandinsky.models.kandisky import KandinskyPipeline
from utility.minio import cmd
from utility.path import separate_bucket_and_file_path


API_URL="http://192.168.3.1:8111"
OUTPUT_PATH="environmental/output/rag_diffusion"

def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--minio-access-key', type=str, help='Minio access key')
        parser.add_argument('--minio-secret-key', type=str, help='Minio secret key')
        parser.add_argument('--images-folder', type=str, help='Folder of image to generate')
        parser.add_argument('--k-nearest-images', type=int, default=5, help='Number of nearest images to use vae latents from')
        parser.add_argument('--dataset', type=str, default="environmental", help='Number of nearest images to use vae latents from')
        parser.add_argument('--decoder-steps', type=int, default=40)
        parser.add_argument('--strength', type=float, default=0.75)
        parser.add_argument('--guidance', type=int, default=12)
        return parser.parse_args()

def load_image_latents(minio_client, file_path:str):
    # get path of clip and vae latents
    bucket, image_path= separate_bucket_and_file_path(file_path)
    vae_latent_path= image_path.replace(".jpg", "_vae_latent.msgpack")
    clip_path= image_path.replace(".jpg", "_clip_kandinsky.msgpack")

    # fetch latents from MinIO
    try:
        vae_data = minio_client.get_object(bucket, vae_latent_path).data
        clip_data = minio_client.get_object(bucket, clip_path).data
        vae_latent_msgpack = msgpack.unpackb(vae_data)
        clip_latent_msgpack = msgpack.unpackb(clip_data)

        vae_latent = torch.tensor(vae_latent_msgpack["latent_vector"])
        clip_vector = torch.tensor(clip_latent_msgpack["clip-feature-vector"])

        image_latents={
            "image_path": image_path,
            "vae_latent": vae_latent,
            "clip_vector": clip_vector
        }

    except Exception as e:
        print(f"Error processing data at path {image_path}: {e}")
        return None

    return image_latents

def load_clip_vae_latents(minio_client, dataset):
    print(f"Fetching clip and vae vectors for all images in the {dataset} dataset")

    response = requests.get(f'{API_URL}/image/get-random-image-list-v1?dataset={dataset}&size=10000')
    # get the list of jobs
    jobs = json.loads(response.content)["response"]
    # get the list of file paths to each image
    image_paths=[job['task_output_file_dict'].get('output_file_path') for job in jobs]
     
    image_latents=[] 
    # load vae and clip vectors for each one
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks and store futures in a dictionary
        futures = [executor.submit(load_image_latents, minio_client, path) for path in image_paths]

        # Use tqdm for progress bar with as_completed
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading vae latents and clip vectors"):
            try:
                result = future.result()
                # Process the result if needed
                if result:
                    image_latents.append(result)
                else:
                   print(f"Error processing job")

            except Exception as exc:
                # Handle the exception (e.g., log it)
                print(f"Error processing job")

    return image_latents

def create_comparison_image(original_images, zeroed_vae_images, rag_diffusion_images):
    # Check if the input lists have the same length
    if not (len(original_images) == len(zeroed_vae_images) == len(rag_diffusion_images)):
        raise ValueError("All input lists must have the same length")

    # Calculate the dimensions for the output image
    image_count = len(original_images)
    if image_count == 0:
        raise ValueError("Input lists must not be empty")
        
    image_width, image_height = original_images[0].size
    margin = 20  # margin between images
    label_height = 40  # height of the text labels
    canvas_width = 3 * image_width + 4 * margin  # 3 columns and 4 margins
    canvas_height = image_count * image_height + (image_count + 1) * margin + label_height  # N rows, N+1 margins, and label height

    # Create a white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Set the font size and style
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # You can change the font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Define the column labels
    labels = ["Original Image", "Generated with Zeroed VAE", "RAG Diffusion"]

    # Draw the labels at the top of each column
    for i, label in enumerate(labels):
        text_width, text_height = draw.textsize(label, font=font)
        x = margin + i * (image_width + margin) + (image_width - text_width) // 2
        y = margin
        draw.text((x, y), label, fill="black", font=font)

    # Paste images on the canvas below the labels
    for i in range(image_count):
        y_offset = i * image_height + (i + 1) * margin + label_height

        # Original image
        canvas.paste(original_images[i], (margin, y_offset))

        # Generated with zeroed VAE
        canvas.paste(zeroed_vae_images[i], (2 * margin + image_width, y_offset))

        # RAG diffusion
        canvas.paste(rag_diffusion_images[i], (3 * margin + 2 * image_width, y_offset))

    return canvas

class RAGInferencePipeline:
    def __init__(self,
                 minio_access_key,
                 minio_secret_key,
                 dataset,
                 k_nearest_images,
                 decoder_steps,
                 guidance,
                 strength):
        
        # get minio client
        self.minio_client = cmd.get_minio_client(minio_access_key,
                                                minio_secret_key)
        self.dataset= dataset
        self.k_nearest_images= k_nearest_images
        
        # get device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = torch.device(device)

        # load image generator
        self.image_generator = KandinskyPipeline(
            device=self.device,
            width= 512,
            height= 512,
            batch_size=1,
            decoder_steps= decoder_steps,
            strength= strength,
            decoder_guidance_scale= guidance
        )

        # load models
        self.image_generator.load_models(task_type="img2img")

        # load clip model
        self.clip = KandinskyCLIPImageEncoder(device= self.device)
        self.clip.load_submodels()
        # initialise list of clip vectors
        self.image_latents=[]

    def initialize_faiss_index(self):
        self.image_latents= load_clip_vae_latents(self.minio_client, self.dataset)
        clip_vectors=[image['clip_vector'].squeeze() for image in self.image_latents]

        dimension = clip_vectors[0].size(0)
        faiss_index = faiss.IndexFlatL2(dimension)
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

        # Convert all_nodes to a contiguous array of float32, required by FAISS
        clip_vectors = torch.stack(clip_vectors).cpu().numpy().astype('float32')
        print(len(clip_vectors), clip_vectors[0])
        faiss_index.add(clip_vectors)

        return faiss_index

    def get_nearest_vectors(self, faiss_index, clip_vectors):
        # Convert vectors to numpy float32 array
        clip_vectors = clip_vectors.cpu().numpy().astype('float32')
        # Compute distances to all vectors in the datasrt and get k nearest neighbors
        _, indices = faiss_index.search(clip_vectors, self.k_nearest_images)  # Find the nearest vectors

        return indices
    
    def generate_images_with_RAG(self, folder_path):
        # get each image and calculate its clip vector
        images=[]
        initial_image=Image.open("./test/test_inpainting/white_512x512.jpg")
        seed= random.seed(time.time())
        seed = random.randint(0, 2 ** 24 - 1)

        faiss_index= self.initialize_faiss_index()
        for filename in os.listdir(folder_path):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)

            try:
                # Open the image file
                img = Image.open(file_path)
                # Optionally, you might want to convert images to a consistent format
                img = img.convert('RGB')
                # Append the image object to the list
                images.append(img)
                print(f"Loaded image: {filename}")
            except IOError:
                print(f"Error opening image: {filename}")
        
        # calculate clip vectors
        image_clip_vectors=[]
        for image in images:
            image_clip_vectors.append(self.clip.get_image_features(image))
        
        image_clip_vectors= torch.stack(image_clip_vectors)

        # get nearest vectors
        nearest_indices= self.get_nearest_vectors(faiss_index, image_clip_vectors)

        original_images=[]
        blank_vae_images=[]
        rag_diffusion_images=[]

        # get initial vae for each image
        for index, image_indices in enumerate(nearest_indices):
            init_vae_latents= torch.stack([self.image_latents[index] for index in image_indices])
            init_vae_latent= torch.mean(init_vae_latents)

            original_image= images[index] 
            original_images.append(original_image)

            generated_image, _= self.image_generator.generate_img2img(init_img=initial_image,
                                                  image_embeds= image_clip_vectors[index],
                                                  seed= seed)
            blank_vae_images.append(generated_image)

            rag_diffusion_image, _= self.image_generator.generate_img2img(init_img=initial_image,
                                                  image_embeds= image_clip_vectors[index],
                                                  init_vae= init_vae_latent,
                                                  seed= seed)
            rag_diffusion_images.append(rag_diffusion_image)
        
        canvas = create_comparison_image(original_images, blank_vae_images, rag_diffusion_images)

        img_byte_arr = io.BytesIO()
        canvas.save(img_byte_arr, format="png")
        img_byte_arr.seek(0)  # Move to the start of the byte array

        cmd.upload_data(self.minio_client, 'datasets', OUTPUT_PATH + f"/inference_comparisons.png" , img_byte_arr)

def main():
    args= parse_args()

    inference_pipeline= RAGInferencePipeline(minio_access_key= args.minio_access_key,
                                             minio_secret_key= args.minio_secret_key,
                                             dataset= args.dataset,
                                             k_nearest_images= args.k_nearest_images,
                                             decoder_steps= args.decoder_steps,
                                             guidance= args.guidance,
                                             strength= args.strength)
    
    inference_pipeline.generate_images_with_RAG(args.images_folder)

if __name__ == "__main__":
    main()