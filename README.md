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
python3 ./worker/worker.py
```