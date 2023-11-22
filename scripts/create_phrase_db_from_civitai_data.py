import csv
import os
import sys
import json
import argparse
import time
from tqdm import tqdm
import re
import json
import csv
import tiktoken


def remove_unwanted_chars(phrase: str):
    # remove non ascii
    phrase_encode = phrase.encode("ascii", "ignore")
    phrase = phrase_encode.decode()

    # remove :n.n
    phrase = re.sub('(:\d+.\d+)|(:)', '', phrase)
    # remove parenthesis
    phrase = re.sub(r'[()]', '', phrase)
    # remove brackets
    phrase = re.sub(r'[\[\]]', '', phrase)
    # remove braces
    phrase = re.sub(r'[{}]', '', phrase)
    # remove quotations
    phrase = re.sub(r'[\"\']', '', phrase)
    # remove slashes
    phrase = re.sub(r'[\\\/]', '', phrase)
    # remove or character
    phrase = re.sub(r'[\|]', '', phrase)
    # remove dash
    phrase = re.sub(r'[\-]', '', phrase)
    # remove underscores
    phrase = re.sub(r'[\_]', '', phrase)
    # remove weird ints and floats
    phrase = re.sub('(\d+.\d+)|(\d+)', '', phrase)
    # remove tabs
    phrase = re.sub(r'[\t]', '', phrase)
    # remove newlines
    phrase = re.sub(r'[\n]', '', phrase)
    # remove carriage returns
    phrase = re.sub(r'[\r]', '', phrase)
    # remove double space
    phrase = re.sub(r'\s{2,}', ' ', phrase)

    return phrase


def process_civitai_data(data_json, output_csv, include_nsfw):
    # initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    # Opening JSON file
    f = open(data_json)
    json_content = json.load(f)

    phrase_id = {}
    phrase_total_count = {}
    phrase_positive_count = {}
    phrase_negative_count = {}

    i = 0
    x = 0
    print("Processing all items in json...")
    for item in tqdm(json_content):
        if item["meta"] is None:
            continue

        if not include_nsfw and item["nsfw"] is True:
            continue

        if "prompt" in item["meta"]:
            positive_prompt = item["meta"]["prompt"]
            i += 1

        if "negativePrompt" in item["meta"]:
            negative_prompt = item["meta"]["negativePrompt"]
            x += 1

        positive_prompt = positive_prompt.replace('\n', ',')
        positive_prompt = positive_prompt.replace(' , ', ',')
        positive_prompt = positive_prompt.replace(' ,', ',')
        positive_prompt = positive_prompt.replace(', ', ',')
        positive_phrases = positive_prompt.split(',')

        negative_prompt = negative_prompt.replace('\n', ',')
        negative_prompt = negative_prompt.replace(' , ', ',')
        negative_prompt = negative_prompt.replace(' ,', ',')
        negative_prompt = negative_prompt.replace(', ', ',')
        negative_phrases = negative_prompt.split(',')

        # positive
        for phrase in positive_phrases:
            # lowercase
            phrase = phrase.lower()
            # remove leading and trailing space
            phrase = phrase.strip()

            if phrase.count('(') != phrase.count(')'):
                continue

            if "<" in phrase or ">" in phrase:
                continue

            if "lora" in phrase:
                continue

            # remove other unwanted chars
            phrase = remove_unwanted_chars(phrase)

            if phrase == '' or phrase.isspace():
                continue

            if phrase not in phrase_id:
                phrase_id[phrase] = len(phrase_id)

            if phrase in phrase_total_count:
                count = phrase_total_count[phrase]
                count += 1
                phrase_total_count[phrase] = count
            else:
                phrase_total_count[phrase] = 1

            if phrase in phrase_positive_count:
                count = phrase_positive_count[phrase]
                count += 1
                phrase_positive_count[phrase] = count
            else:
                phrase_positive_count[phrase] = 1

        # negative
        for phrase in negative_phrases:
            # lowercase
            phrase = phrase.lower()
            # remove leading and trailing space
            phrase = phrase.strip()

            if phrase.count('(') != phrase.count(')'):
                continue

            if "<" in phrase or ">" in phrase:
                continue

            if "lora" in phrase:
                continue

            # remove other unwanted chars
            phrase = remove_unwanted_chars(phrase)

            if phrase == '' or phrase.isspace():
                continue

            if phrase not in phrase_id:
                phrase_id[phrase] = len(phrase_id)

            if phrase in phrase_total_count:
                count = phrase_total_count[phrase]
                count += 1
                phrase_total_count[phrase] = count
            else:
                phrase_total_count[phrase] = 1

            if phrase in phrase_negative_count:
                count = phrase_negative_count[phrase]
                count += 1
                phrase_negative_count[phrase] = count
            else:
                phrase_negative_count[phrase] = 1

    print("Total positive prompts: {}".format(i))
    print("Total neg prompts: {}".format(x))
    print("Total phrases: {}".format(len(phrase_id)))

    # sort
    sorted_dict = {}
    sorted_keys = sorted(phrase_total_count, key=phrase_total_count.get, reverse=True)
    for w in sorted_keys:
        sorted_dict[w] = phrase_total_count[w]

    # save csv
    # field names
    fields = ['index', 'total count', 'positive count', 'negative count', 'token size', 'phrase str']

    # writing to csv file
    with open(output_csv, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        for key, val in sorted_dict.items():
            if key == "":
                continue

            # compute token size
            phrase_tokens = enc.encode(key, allowed_special={'<|endoftext|>'})
            num_tokens = len(phrase_tokens)
            positive_count = phrase_positive_count[key] if key in phrase_positive_count else 0
            negative_count = phrase_negative_count[key] if key in phrase_negative_count else 0
            row = [phrase_id[key], val, positive_count, negative_count, num_tokens, key]
            csvwriter.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Tool to clean csv phrase database")

    # Required parameters
    parser.add_argument("--source", type=str,
                        help="Source civitai json data")
    parser.add_argument("--output", type=str,
                        help="Path to save the new csv")
    parser.add_argument("--include-nsfw", type=bool, default=False,
                        help="Option to include nsfw prompts")

    return parser.parse_args()


def main():
    args = parse_args()

    process_civitai_data(args.source, args.output, args.include_nsfw)


if __name__ == '__main__':
    main()