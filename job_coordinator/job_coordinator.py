
import argparse
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts

def parse_args():
    parser = argparse.ArgumentParser(description="generate prompts")

    # Required parameters
    parser.add_argument("--base_prompts_path", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    csv_dataset_path = 'input/civitai_phrases_database_v6.csv'
    prompt_count = 5,
    base_prompts_csv_path = 'input/base-prompts/icon/base_prompts_icon_1.csv',
    dataset_name = 'icons',
    csv_phrase_limit = 0,
    positive_prefix = "",
    init_img_path = "./test/test_inpainting/white_512x512.jpg",
    mask_path = "./test/test_inpainting/icon_mask.png"


    generate_inpainting_generation_jobs_using_generated_prompts_and_base_prompts(
        csv_dataset_path,
        prompt_count,
        base_prompts_csv_path,
        dataset_name,
        csv_phrase_limit,
        positive_prefix,
        init_img_path,
        mask_path
    )


if __name__ == '__main__':
    main()