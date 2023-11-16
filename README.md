# kcg-ml-image-pipeline

- [kcg-ml-image-pipeline](#kcg-ml-image-pipeline)
  - [Prerequisites](#prerequisites)
    - [Use Python Environment:](#use-python-environment)
    - [Install Dependencies](#install-dependencies)
    - [Install MongoDB and Running MongoDB](#install-mongodb-and-running-mongodb)
  - [Orchestration API](#orchestration-api)
  - [Worker](#worker)
    - [Prompt Generator](#prompt-generator)
  - [Model Training](#model-training)
    - [AB Ranking Linear Model](#ab-ranking-linear-model)
    - [AB Ranking Efficient Net Model](#ab-ranking-efficient-net-model)

## Prerequisites
### Use Python Environment:
Create an env by running:

    python3 -m venv venv

Then activate it by running:
    
    source ./venv/bin/activate

### Install Dependencies
Install dependencies by running:

    pip install -r requirements.txt

### Install MongoDB and Running MongoDB
For MacOS:
```
    brew install mongodb-community
    mkdir /data/db
    brew services start mongodb-community
```

For Ubuntu:
```
    sudo apt install mongodb
    mkdir /data
    mkdir /data/db
    sudo systemctl start mongodb
    sudo systemctl enable mongodb
```

For Devboxes (Containers):
```
    sudo apt install mongodb
    mkdir /data
    mkdir /data/db
    sudo systemctl enable mongodb
    mongod &> mongodb.log &
```


## Orchestration API
To deploy orchestration api, run:

    uvicorn orchestration.api.main:app --reload
    
or

    python3 main.py

API Docs should be accessible at:

    http://127.0.0.1:8000/docs


## Worker
To run the worker, run:
```
options:
  -h, --help            show this help message and exit
  
  --device DEVICE
                Choose a device for the worker example ('cuda:0', 'cuda:1' ..)
```
```
python3 ./worker/worker.py --device cuda:0
```


### Prompt Generator

Generates prompts and saves to a zip file

```
usage: prompt_generator.py [-h] [--positive-prefix POSITIVE_PREFIX] [--num-prompts NUM_PROMPTS] [--csv-phrase-limit CSV_PHRASE_LIMIT] [--csv-path CSV_PATH] [--output OUTPUT] [--positive-ratio-threshold POSITIVE_RATIO_THRESHOLD] [--negative-ratio-threshold NEGATIVE_RATIO_THRESHOLD]
                           [--use-threshold USE_THRESHOLD] [--proportional-selection PROPORTIONAL_SELECTION]

Prompt Generator CLI tool generates prompts from phrases inside a csv

options:
  -h, --help            show this help message and exit
  --positive-prefix POSITIVE_PREFIX
                        Prefix phrase to add to positive prompts
  --num-prompts NUM_PROMPTS
                        Number of prompts to generate
  --csv-phrase-limit CSV_PHRASE_LIMIT
                        Number of phrases to use from the csv data
  --csv-path CSV_PATH   Full path to the csv path
  --output OUTPUT       Output path for dataset zip containing prompt list npz
  --positive-ratio-threshold POSITIVE_RATIO_THRESHOLD
                        Threshold ratio of positive/negative to use a phrase for positive prompt
  --negative-ratio-threshold NEGATIVE_RATIO_THRESHOLD
                        Threshold ratio of negative/positive to use a phrase for negative prompt
  --use-threshold USE_THRESHOLD
                        True if positive and negative ratio will be used
  --proportional-selection PROPORTIONAL_SELECTION
                        True if proportional selection will be used to get the phrases
```

```
To get civitai csv, download via:
https://mega.nz/file/wNIXHKDA#CIB92fLkCquatWvzUAjjcKbrKCUmO2ffRVrzE3XYQVM
```
Example Usage:

```
python ./prompt_generation/scripts/prompt_generator.py --num-prompts 10000 --positive-prefix "icon, game icon, crystal, high resolution, contour, game icon, jewels, minerals, stones, gems, flat, vector art, game art, stylized, cell shaded, 8bit, 16bit, retro, russian futurism" --csv-phrase-limit 512 --csv-path ./input/civitai_phrases_database_v6.csv --output ./output/prompt_list_civitai_10000 --positive-ratio-threshold 3 --negative-ratio-threshold 3
```

## Model Training
### AB Ranking Linear Model
To train ab ranking linear model:
1. Install dependencies by running `pip install -r requirements.txt`
2. Run using `python ./scripts/ab_ranking_linear.py --minio-access-key access --minio-secret-key secret --dataset-name propaganda-poster --input-type embedding`

### AB Ranking ELM Model
To train ab ranking linear model:
1. Install dependencies by running `pip install -r requirements.txt`
2. Run using `python ./scripts/ab_ranking_elm_v1.py --minio-access-key access --minio-secret-key secret --dataset-name propaganda-poster --input-type embedding`

### AB Ranking Efficient Net Model
To train ab ranking linear model:
1. Install dependencies by running `pip install -r requirements.txt`
2. Go to [ab_ranking_efficient_net test run entrypoint](https://github.com/kk-digital/kcg-ml-image-pipeline/blob/main/training_worker/ab_ranking/script/ab_ranking_efficient_net.py#L184)
2. Input the minio server's address, your minio secret, and your minio access key.
3. Run using `python python ./training_worker/ab_ranking/script/ab_ranking_efficient_net.py`

## Tools
### Check Dataset Clip
This checks clip data for selection datapoints in the dataset and creates a clip calculation task if clip data doesn't exist.
```
python ./scripts/check_dataset_clip.py --minio-access-key nkjYl5jO4QnpxQU0k0M1 --minio-secret-key MYtmJ9jhdlyYx3T1McYy4Z0HB3FkxjmITXLEPKA1 --dataset-name all
```

### Generate dataset for addition / removal prompt mutator
```
python scripts/generate_prompt_mutator_dataset.py \
    --minio-access-key <access key> \
    --minio-secret-key <secret key> \
    --df_phrase_path <CSV to sample phrases, must have "phrase str" column> \
    --df_seed_path <CSV to get seed prompt, must have "positive_prompt" column> \ 
    --n_data <number of data samples to generate> \
    --minio_upload_path <minio folder path>
```