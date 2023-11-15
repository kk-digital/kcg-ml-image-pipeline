import os
import msgpack
import argparse
import tqdm

import pandas as pd

from transformers import CLIPTokenizer

def main(
    tokenizer_path,
    embedding_path,
    save_path
):
    # load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    # get file paths for embedding msgpack
    embedding_file_paths = []
    for root, dirs, files in os.walk(embedding_path):
        for file in files:
            if file.endswith('_embedding.msgpack'):
                embedding_file_paths.append(os.path.join(root, file))
    embedding_file_paths = sorted(embedding_file_paths)

    print('Reading embedding msgpack files and computing lengths')
    # initialize list to store data that exceeds 77
    anomalies = []
    for file_path in tqdm.tqdm(embedding_file_paths):
        # read msgpack
        with open(file_path, 'rb') as f:
            data = f.read()

        data = msgpack.unpackb(data)

        # get prompts
        positive_prompt = data['positive_prompt']
        negative_prompt = data['negative_prompt']

        # compute token lengths
        # the lengths include start and end tokens
        positive_length = tokenizer(positive_prompt, return_length=True)['length']
        # someetimes negative prompt may be NaN
        # if NaN, set length as 0
        if isinstance(negative_prompt, str):
            negative_length = tokenizer(negative_prompt, return_length=True)['length']
        else:
            negative_length = 0

        # if positive / negative prompt lengths exceed 77, record it
        if positive_length > 77 or negative_length > 77:
            data['positive_length'] = positive_length
            data['negative_length'] = negative_length
            anomalies.append(data)

    # convert to dataframe and save as csv
    df_anomaly = pd.DataFrame(anomalies)
    df_anomaly.to_csv(save_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_path', type=str, help='Directory path to tokenizer')
    ap.add_argument('--embedding_path', type=str, help='Directory path to embedding msgpacks')
    ap.add_argument('--save_path', type=str, help='File path to save output csv')
    args = ap.parse_args()

    main(
        tokenizer_path=args.tokenizer_path,
        embedding_path=args.embedding_path,
        save_path=args.save_path
    )