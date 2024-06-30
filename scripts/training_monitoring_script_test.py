import json
import random









# ADDED BY ME
from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from io import BytesIO
import io
import os
import sys
base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from data_loader.ab_ranking_dataset_loader import ABRankingDatasetLoader
from utility.minio import cmd
from utility.clip.clip_text_embedder import tensor_attention_pooling
import urllib.request
from urllib.error import HTTPError
from torch.utils.data import random_split, DataLoader
from PIL import Image
import requests
import msgpack 
from kandinsky.models.clip_image_encoder.clip_image_encoder import KandinskyCLIPImageEncoder
import tempfile
import csv
import pandas as pd
from torch.utils.data import ConcatDataset
import argparse
from safetensors.torch import load_model, save_model
from training_worker.classifiers.models.reports.get_model_card import get_model_card_buf
from os.path import basename





# Simulating loss values
num_epochs = 5
batches_per_epoch = 10
training_logs = []

for epoch in range(num_epochs):
    epoch_log = {"epoch": epoch + 1, "batch_losses": [], "epoch_loss": 0}
    epoch_loss = 0

    for batch_idx in range(batches_per_epoch):
        batch_loss = random.uniform(0.1, 1.0)  # Simulating a random loss value
        epoch_log["batch_losses"].append({"batch": batch_idx + 1, "loss": batch_loss})
        epoch_loss += batch_loss

    epoch_log["epoch_loss"] = epoch_loss / batches_per_epoch
    training_logs.append(epoch_log)

# Save logs to JSON file
log_file = "training_log.json"
with open(log_file, "w") as f:
    json.dump(training_logs, f, indent=4)


minio_client = cmd.get_minio_client("D6ybtPLyUrca5IdZfCIM",
            "2LZ6pqIGOiZGcjPTR6DZPlElWBkRTkaLkyLIBt4V"
            )


bucket_name = "datasets/environmental/output/my_tests"
object_name = "training_log.json"

# # Ensure the bucket exists
# found = minio_client.bucket_exists(bucket_name)
# if not found:
#     minio_client.make_bucket(bucket_name)
# else:
#     print(f"Bucket '{bucket_name}' already exists.")

# Upload the JSON file to MinIO
# try:
#     minio_client.fput_object(bucket_name, object_name, log_file)
#     print(f"'{log_file}' is successfully uploaded as '{object_name}' to bucket '{bucket_name}'.")
# except S3Error as e:
#     print(f"Error occurred: {e}")

# Clean up the local JSON file if needed
os.remove(log_file)