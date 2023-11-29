import csv
import requests
import argparse

def upload_jsons_from_csv(csv_file_path):
    api_url = "http://123.176.98.90:8764"

    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            job_uuid_1 = row["job_uuid_1"]
            job_uuid_2 = row["job_uuid_2"]
            policy = row["policy"]

            # Endpoint URL with job_uuid_1, job_uuid_2, and policy as query parameters
            endpoint_url = f"{api_url}/ranking-queue/add-image-pair-to-queue?job_uuid_1={job_uuid_1}&job_uuid_2={job_uuid_2}&policy={policy}"
            response = requests.post(endpoint_url)

            if response.status_code == 200:
                print(f"Successfully processed job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}")
            else:
                print(f"Failed to process job pair: UUID1: {job_uuid_1}, UUID2: {job_uuid_2}. Response: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Upload JSONs from CSV to MinIO via API")
    parser.add_argument("--csv_filepath", type=str, required=True, help="Path to the CSV file")
    
    args = parser.parse_args()
    csv_file_path = args.csv_filepath
    
    upload_jsons_from_csv(csv_file_path)

if __name__ == "__main__":
    main()
