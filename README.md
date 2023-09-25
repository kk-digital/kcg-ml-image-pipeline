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
    brew services start mongodb-community
```

For Ubuntu:
```
    sudo apt install mongodb
    sudo systemctl start mongodb
    sudo systemctl enable mongodb
```
## Orchestration API
To deploy orchestration api, run:

    uvicorn orchestration.api.main:app --reload

API Docs should be accessible at:

    http://127.0.0.1:8000/docs