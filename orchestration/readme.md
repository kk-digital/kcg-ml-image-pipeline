# Usage Example of Endpoint for Devbox


### Step 1: Connect to Devbox

1. Open your terminal and connect to your Devbox.
    ```bash
    ssh root@103.20.60.90 -p <port>
    ```

### Step 2: Clone the Repository

1. If you do not have the repository on your Devbox, clone it using the following command:

    ```bash
    cd /devbox
    proxychains git clone https://<github-username>:<personal-access-token>@github.com/kk-digital/kcg-ml-image-pipeline.git
    ```

2. If you already have the repository on your Devbox, navigate to the repository directory:
    ```bash
    cd /devbox/kcg-ml-image-pipeline
    ```

### Step 3: Update the Repository

1. If there are any changes in the repository, pull the latest changes:
    ```bash
    proxychains git pull
    ```

### Step 4: Install Dependencies

1. Install the required Python for orchestration:
    ```bash
    proxychains pip install -r requirements_orchestration.txt
    ```

### Step 5: Run the Application

1. Start the application:
    ```bash
    python3 main.py
    ```

### Step 6: Connect to Devbox from Another Terminal Window

1. Open a new terminal window and connect to the same Devbox:
    ```bash
    ssh root@103.20.60.90 -p <port>
    ```

### Step 7: Use the Curl Command to Run the Endpoint

1. Use the `curl` command to interact with the endpoint. Here is an example:
    ```bash
    curl -X 'GET' \
      'http://localhost:8000/queue/image-generation/list-completed-jobs?dataset=waifu&limit=10' \
      -H 'accept: application/json'
    ```
