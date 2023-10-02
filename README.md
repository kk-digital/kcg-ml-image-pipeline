# kcg-ml-image-pipeline

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


