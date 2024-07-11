from io import BytesIO
import os
import sys
from minio import Minio
import msgpack
from tqdm import tqdm

base_dir = "./"
sys.path.insert(0, base_dir)
sys.path.insert(0, os.getcwd())

class ImageDatasetLoader:
    def __init__(self,
                 minio_client: Minio,
                 bucket: str, 
                 dataset:str):
        
        self.minio_client= minio_client
        self.bucket = bucket
        self.dataset = dataset

    # function for loading all clip vectors for images
    def load_dataset(self):
        latents_directory= f"{self.dataset}/clip_vectors/"

        # get list of latent batches
        latent_batches= self.minio_client.list_objects(bucket_name=self.bucket, prefix=latents_directory)
        image_features= []

        print("Loading vae and clip batches.......")
        for batch in tqdm(latent_batches):
            try:
                response = self.minio_client.get_object(self.bucket, batch.object_name)
                data = BytesIO(response.read())
                data= msgpack.unpackb(data.getvalue(), raw=False)
                image_features.extend(data)
            except Exception as e:
                print(f"Failed to retrieve or parse the file: {e}")

        return image_features