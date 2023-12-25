
import sys

base_directory = "./"
sys.path.insert(0, base_directory)

from worker.prompt_generation.prompt_generator import (generate_inpainting_job_with_temperature,
                                                       generate_image_generation_jobs_with_temperature)
def generate_icon_generation_jobs(prompt_job_generator_state):

    dataset_name = 'icons'
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/icon_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)

    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_inpainting_job_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        init_img_path=init_img_path,
        mask_path=mask_path,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k
    )

def generate_character_generation_jobs(prompt_job_generator_state):

    dataset_name = "character"
    init_img_path = "./test/test_inpainting/white_512x512.jpg"
    mask_path = "./test/test_inpainting/character_mask.png"

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)
    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_inpainting_job_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        init_img_path=init_img_path,
        mask_path=mask_path,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k

    )

def generate_propaganda_posters_image_generation_jobs(prompt_job_generator_state):

    dataset_name = 'propaganda-poster'

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_image_generation_jobs_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k
    )


def generate_environmental_image_generation_jobs(prompt_job_generator_state):

    dataset_name = 'environmental'

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_image_generation_jobs_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k
    )

def generate_waifu_image_generation_jobs(prompt_job_generator_state):

    dataset_name = 'waifu'

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_image_generation_jobs_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k
    )


def generate_mechs_image_generation_jobs(prompt_job_generator_state):
    dataset_name = "mech"

    random_mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)

    init_img_path = random_mask['init_image']
    mask_path = random_mask['mask']

    mask = prompt_job_generator_state.get_random_dataset_mask(dataset_name)
    if mask != None:
        init_img_path = mask['init_image']
        mask_path = mask['mask']

    print(f"Adding '{dataset_name}' generation job")

    prompt_queue = prompt_job_generator_state.prompt_queue
    scored_prompt = prompt_queue.get_dataset_prompt(dataset_name)

    if scored_prompt is None:
        return

    positive_prompt = scored_prompt.positive_prompt
    negative_prompt = scored_prompt.negative_prompt
    prompt_scoring_model = scored_prompt.scoring_model
    prompt_score = scored_prompt.score
    prompt_generation_policy = scored_prompt.generation_policy
    top_k = scored_prompt.top_k
    boltzman_temperature = scored_prompt.boltzman_temperature
    boltzman_k = scored_prompt.boltzman_k

    generate_inpainting_job_with_temperature(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        prompt_scoring_model=prompt_scoring_model,
        prompt_score=prompt_score,
        prompt_generation_policy=prompt_generation_policy,
        top_k=top_k,
        dataset_name=dataset_name,
        init_img_path=init_img_path,
        mask_path=mask_path,
        boltzman_temperature=boltzman_temperature,
        boltzman_k=boltzman_k

    )

