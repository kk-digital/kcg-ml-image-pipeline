import pandas as pd
from transformers import CLIPTokenizer
from minio import Minio
import msgpack
from tqdm import tqdm

MAX_LENGTH = 77
MODEL_NAME = 'openai/clip-vit-large-patch14'

def connect_to_minio_client(minio_ip_addr, access_key, secret_key):
    print("Connecting to minio client...")
    client = Minio(minio_ip_addr, access_key, secret_key, secure=False)
    print("Successfully connected to minio client...")
    return client

def process_msgpack_file(client, bucket_name, object_name, tokenizer):
    print(f"Processing file: {object_name}")
    response = client.get_object(bucket_name, object_name)
    file_data = msgpack.unpackb(response.read(), raw=False)
    
    positive_prompt = file_data['positive_prompt']
    negative_prompt = file_data['negative_prompt']

    print(f"Tokenizing prompts in: {object_name}")
    positive_encoding = tokenizer(positive_prompt, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", return_length=True)
    negative_encoding = tokenizer(negative_prompt, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", return_length=True)

    positive_length = positive_encoding['length'].numpy()[0]
    negative_length = negative_encoding['length'].numpy()[0]

    return positive_prompt, positive_length, negative_prompt, negative_length

def main():

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    minio_client = connect_to_minio_client("192.168.3.5:9000", "v048BpXpWrsVIHUfdAix", "4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu")

    bucket_name = 'datasets'
    data = {'prompt': [], 'token_length': [], 'label': []}
    objects = list(minio_client.list_objects(bucket_name, prefix='test-generations/', recursive=True))
    print(f"Found {len(objects)} objects in MinIO bucket.")

    for obj in tqdm(objects, desc="Processing msgpack files"):
        if obj.object_name.endswith('_data.msgpack'):
            positive_prompt, positive_length, negative_prompt, negative_length = process_msgpack_file(minio_client, bucket_name, obj.object_name, tokenizer)
            
            data['prompt'].extend([positive_prompt, negative_prompt])
            data['token_length'].extend([positive_length, negative_length])
            data['label'].extend(['positive', 'negative'])

    df = pd.DataFrame(data)
    csv_file_path = 'environmental_prompt_lengths.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to {csv_file_path}")

if __name__ == "__main__":
    main()
