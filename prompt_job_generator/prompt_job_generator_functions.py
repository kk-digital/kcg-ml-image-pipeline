
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import (generate_inpainting_job,
                                                       generate_image_generation_jobs)
def generate_icon_generation_jobs(prompt_job_generator_state):
    prompt_count = 20
    base_prompts_csv_path = 'input/dataset-config/icon/base-prompts-icon-2.csv'
    dataset_name = 'icons'
    positive_prefix = ""
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/icon_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)

    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    efficient_net_model = prompt_job_generator_state.get_efficient_net_model(dataset_name)

    generate_inpainting_job(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        init_img_path=init_img_path,
        mask_path=mask_path,
        efficient_net_model=efficient_net_model,
        clip_text_embedder=prompt_job_generator_state.clip_text_embedder

    )

def generate_character_generation_jobs(prompt_job_generator_state):
    prompt_count = 20
    base_prompts_csv_path = 'input/dataset-config/character/base-prompts-waifu.csv'
    dataset_name = "character"
    positive_prefix = ""
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/character_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)
    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    efficient_net_model = prompt_job_generator_state.get_efficient_net_model(dataset_name)

    generate_inpainting_job(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        init_img_path=init_img_path,
        mask_path=mask_path,
        efficient_net_model=efficient_net_model,
        clip_text_embedder=prompt_job_generator_state.clip_text_embedder

    )

def generate_propaganda_posters_image_generation_jobs(prompt_job_generator_state):

    prompt_count = 20
    base_prompts_csv_path = 'input/dataset-config/propaganda-poster/base-prompts-propaganda-poster.csv'
    dataset_name = 'propaganda-poster'
    positive_prefix = ""

    print(f"Adding '{dataset_name}' generation job")

    efficient_net_model = prompt_job_generator_state.get_efficient_net_model(dataset_name)

    generate_image_generation_jobs(
        phrases=prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        efficient_net_model=efficient_net_model,
        clip_text_embedder=prompt_job_generator_state.clip_text_embedder
    )

def generate_mechs_image_generation_jobs(prompt_job_generator_state):
    prompt_count = 20
    base_prompts_csv_path = 'input/dataset-config/mech/base-prompts-mechs.csv'
    dataset_name = 'mech'
    positive_prefix = ""

    print(f"Adding '{dataset_name}' generation job")

    efficient_net_model = prompt_job_generator_state.get_efficient_net_model(dataset_name)

    generate_image_generation_jobs(
        phrases = prompt_job_generator_state.phrases,
        phrases_token_size=prompt_job_generator_state.phrases_token_size,
        positive_count_list=prompt_job_generator_state.positive_count_list,
        negative_count_list=prompt_job_generator_state.negative_count_list,
        prompt_count=prompt_count,
        base_prompts_csv_path=base_prompts_csv_path,
        dataset_name=dataset_name,
        positive_prefix=positive_prefix,
        efficient_net_model=efficient_net_model,
        clip_text_embedder=prompt_job_generator_state.clip_text_embedder
    )

