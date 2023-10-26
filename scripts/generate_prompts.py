import argparse
import sys
from datetime import datetime
import csv

base_directory = "./"
sys.path.insert(0, base_directory)


from worker.prompt_generation.prompt_generator import (initialize_prompt_list_from_csv)
from worker.prompt_generation.prompt_generator import generate_prompts_proportional_selection, generate_base_prompts, load_base_prompts
from utility.minio import cmd

def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--prompt_count", type=int, default=1)
    parser.add_argument("--csv_dataset_path", type=str, default='input/civitai_phrases_database_v6.csv')
    parser.add_argument("--csv_base_prompts", type=str,
                        default='')

    parser.add_argument("--minio_access_key", type=str, default='v048BpXpWrsVIHUfdAix')
    parser.add_argument("--minio_secret_key", type=str, default='4TFS20qkxVuX2HaC8ezAgG7GaDlVI1TqSPs0BKyu')

    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()


def main():
    args = parse_args()

    csv_base_prompts = args.csv_base_prompts
    csv_dataset_path = args.csv_dataset_path
    prompt_count = args.prompt_count
    minio_secret_key = args.minio_secret_key
    minio_access_key = args.minio_access_key

    minio_client = cmd.get_minio_client(minio_access_key, minio_secret_key)

    print(f'generating {prompt_count} prompts ')


    phrases, phrases_token_size, positive_count_list, negative_count_list = initialize_prompt_list_from_csv(
        csv_dataset_path, 0)

    batch_size = 1000

    batch_count = int(prompt_count / batch_size)

    if csv_base_prompts != '' and csv_base_prompts is not None:
        base_prompt_population = load_base_prompts(csv_base_prompts)
    else:
        base_prompt_population = None

    # Specify the CSV file name
    txt_file = "prompt_list.txt"

    # Open the CSV file in write mode
    # Open the file in write mode and write each string on a new line
    with open(txt_file, 'w') as file:
        for i in range(0, batch_count):
            begin_time = datetime.now()
            prompts = generate_prompts_proportional_selection(phrases,
                                                              phrases_token_size,
                                                              positive_count_list,
                                                              negative_count_list,
                                                              batch_size,
                                                              '')

            prompt_list = []
            for index in range(0, batch_size):

                prompt = prompts[index]
                # N Base Prompt Phrases
                # Hard coded probability of choose 0,1,2,3,4,5, etc base prompt phrases
                # Chance for 0 base prompt phrases should be 30%
                # choose_probability = [0.3, 0.3, 0.2, 0.2, 0.2]
                choose_probability = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

                if base_prompt_population is not None:
                    base_prompt_list = generate_base_prompts(base_prompt_population, choose_probability)
                else:
                    base_prompt_list = []

                base_prompts = ''

                for base_prompt in base_prompt_list:
                    base_prompts = base_prompts + base_prompt + ', '

                positive_text_prompt = base_prompts + prompt.positive_prompt_str

                prompt_list.append(positive_text_prompt)

            end_time = datetime.now()
            elapsed_time = end_time - begin_time

            print(f"Execution time: {elapsed_time}")

            for item in prompt_list:
                file.write(item + "\n")

    cmd.upload_from_file(minio_client, 'datasets', 'prompts.txt', txt_file)

if __name__ == '__main__':
    main()



