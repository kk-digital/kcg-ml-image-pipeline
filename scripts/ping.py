import requests

def check_http_connection(url: str) -> None:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"Successfully connected to {url}")
        else:
            print(f"Failed to connect to {url}, status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"An error occurred while trying to connect to {url}: {e}")

# Example usage
if __name__ == "__main__":
    url_to_check = "http://103.20.60.90:9001"  
    check_http_connection(url_to_check)
