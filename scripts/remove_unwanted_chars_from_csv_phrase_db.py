import csv
import os
import sys
import json
import argparse
import time
from tqdm import tqdm
import re

def remove_unwanted_chars(phrase: str):
    # remove non ascii
    phrase_encode = phrase.encode("ascii", "ignore")
    phrase = phrase_encode.decode()

    # remove :n.n
    phrase = re.sub('(:\d+.\d+)|(:)', '', phrase)
    # remove parenthesis, brackets, braces, parenthesis, quotation, slashes, or, dash, underscores
    phrase = re.sub(r'[()\[\]{}\"\'\\\/\|\-\_]', '', phrase)

    return phrase

def process_csv(source_csv, output_path):
    new_csv = open(output_path, mode='w')
    new_csv_writer = csv.writer(new_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    with open(source_csv) as source_csv:
        source_csv_reader = csv.reader(source_csv, delimiter=',')
        line_count = 0
        for row in tqdm(source_csv_reader):
            if line_count == 0:
                new_csv_writer.writerow(row)

                line_count += 1
            else:
                new_phrase = remove_unwanted_chars(row[5])
                row[5] = new_phrase
                if new_phrase != '' and not new_phrase.isspace():
                    new_csv_writer.writerow(row)

                line_count += 1

        print(f'Processed {line_count} lines.')

def parse_args():
    parser = argparse.ArgumentParser(description="Tool to clean csv phrase database")

    # Required parameters
    parser.add_argument("--source", type=str,
                        help="Source csv path")
    parser.add_argument("--output", type=str,
                        help="Path to save the new csv")

    return parser.parse_args()

def main():
    args = parse_args()

    process_csv(args.source, args.output)


if __name__ == '__main__':
    main()