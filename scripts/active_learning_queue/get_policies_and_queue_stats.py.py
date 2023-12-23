import requests
import json

def main():
    # Base URL of your FastAPI application
    base_url = "http://123.176.98.90:8764"  

    # Endpoint URLs
    list_policies_url = f"{base_url}/active-learning-policy/list-policies"
    count_queue_pairs_url = f"{base_url}/active-learning-queue/count-queue-pairs"

    # Making API calls
    policies_response = requests.get(list_policies_url)
    queue_count_response = requests.get(count_queue_pairs_url)

    if policies_response.status_code == 200 and queue_count_response.status_code == 200:
        # Extract data from responses
        policies = policies_response.json()
        queue_count = queue_count_response.json()

        # Combine the results into a single JSON object
        combined_results = {
            "active_learning_policies": policies,
            "number_of_items_in_queue": queue_count
        }

        # Print the combined results as pretty-printed JSON
        print(json.dumps(combined_results, indent=4))
    else:
        print("Failed to fetch data from the APIs.")

if __name__ == "__main__":
    main()
