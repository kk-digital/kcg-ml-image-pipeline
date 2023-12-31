import csv
import requests
import argparse
import tempfile
import os

def upload_jsons_from_csv(csv_file_path, progress_file_path=None):
    api_url = "http://123.176.98.90:8764"

    # Use a temporary file if no progress file path is provided
    temp_progress_file = None
    if not progress_file_path:
        temp_progress_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        progress_file_path = temp_progress_file.name

    start_index = 0
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as progress_file:
            progress_content = progress_file.read().strip()
            if progress_content.isdigit():  # Check if the content is a number
                start_index = int(progress_content)

    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if i < start_index:
                continue  # Skip already processed entries

            job_uuid = row["job_uuid"]
            policy = row["policy"]

            endpoint_url = f"{api_url}/ranking-queue/add-image-to-queue?job_uuid={job_uuid}&policy={policy}"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job UUID: {job_uuid}")
                with open(progress_file_path, 'w') as progress_file:
                    progress_file.write(str(i+1))  # Write the next index to be processed
            else:
                print(f"Failed to process job UUID: {job_uuid}. Response: {response.status_code} - {response.text}")

    # Delete the temporary progress file if it was used
    if temp_progress_file:
        os.remove(temp_progress_file.name)

def main():
    parser = argparse.ArgumentParser(description="Upload JSONs from CSV to MinIO via API")
    parser.add_argument("--csv_filepath", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--progress_filepath", type=str, help="Path to the progress file (optional)")

    args = parser.parse_args()
    csv_file_path = args.csv_filepath

    # If progress_filepath is not provided, use default based on csv_file_path
    if args.progress_filepath:
        progress_file_path = args.progress_filepath
    else:
        progress_file_path = f"{csv_file_path}_progress.txt"

    print(f"Progress file path: {progress_file_path}")
    upload_jsons_from_csv(csv_file_path, progress_file_path)

if __name__ == "__main__":
    main()


