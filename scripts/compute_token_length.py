import os
import msgpack
import argparse
import tqdm
import tiktoken

import pandas as pd

from transformers import CLIPTokenizer, CLIPTokenizerFast

def main(
    tokenizer_path,
    embedding_path,
    save_path
):
    # load tokenizer
    openai_tokenizer = CLIPTokenizer.from_pretrained('./input/openai-vit-large-patch14-tokenizer')
    fast_tokenizer = CLIPTokenizerFast.from_pretrained('./input/openai-vit-large-patch14-tokenizer')
    kcg_tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    tt_tokenizer = tiktoken.get_encoding('cl100k_base')

    # get file paths for embedding msgpack
    embedding_file_paths = []
    for root, dirs, files in os.walk(embedding_path):
        for file in files:
            if file.endswith('_embedding.msgpack'):
                embedding_file_paths.append(os.path.join(root, file))
    embedding_file_paths = sorted(embedding_file_paths)

    print('Reading embedding msgpack files and computing lengths')
    # initialize list to store data
    dataset = []
    for file_path in tqdm.tqdm(embedding_file_paths):
        # read msgpack
        with open(file_path, 'rb') as f:
            data = f.read()

        data = msgpack.unpackb(data)
        # delete embedding from dict to reduce memory
        del data['positive_embedding']
        del data['negative_embedding']

        # get prompts
        positive_prompt = data['positive_prompt']
        negative_prompt = data['negative_prompt']

        # compute token lengths
        positive_openai_length = openai_tokenizer(positive_prompt, return_length=True)['length']
        positive_fast_length = fast_tokenizer(positive_prompt, return_length=True)['length']
        positive_kcg_length = kcg_tokenizer(positive_prompt, return_length=True)['length']
        positive_tt_length = len(tt_tokenizer.encode(positive_prompt))
        
        # sometimes negative prompt may be NaN
        # if NaN, set length as 0
        if isinstance(negative_prompt, str):
            negative_openai_length = openai_tokenizer(negative_prompt, return_length=True)['length']
            negative_fast_length = fast_tokenizer(negative_prompt, return_length=True)['length']
            negative_kcg_length = kcg_tokenizer(negative_prompt, return_length=True)['length']
            negative_tt_length = len(tt_tokenizer.encode(negative_prompt))
        else:
            negative_openai_length = 0
            negative_fast_length = 0
            negative_kcg_length = 0
            negative_tt_length = 0

        data['positive_openai_length'] = positive_openai_length
        data['positive_fast_length'] = positive_fast_length
        data['positive_kcg_length'] = positive_kcg_length
        data['positive_tt_length'] = positive_tt_length

        data['negative_openai_length'] = negative_openai_length
        data['negative_fast_length'] = negative_fast_length
        data['negative_kcg_length'] = negative_kcg_length
        data['negative_tt_length'] = negative_tt_length

        data['openai=kcg (pos)'] = (positive_openai_length == positive_kcg_length)
        data['openai=kcg (neg)'] = (negative_openai_length == negative_kcg_length)

        dataset.append(data)

    # convert to dataframe and save as csv
    df_dataset = pd.DataFrame(dataset)
    df_dataset.to_csv(save_path, index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_path', type=str, help='Directory path to kcg tokenizer')
    ap.add_argument('--embedding_path', type=str, help='Directory path to embedding msgpacks')
    ap.add_argument('--save_path', type=str, help='File path to save output csv')
    args = ap.parse_args()

    main(
        tokenizer_path=args.tokenizer_path,
        embedding_path=args.embedding_path,
        save_path=args.save_path
    )