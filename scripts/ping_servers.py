import os
import requests
from ping3 import ping

def check_ping(host):
    """Ping a host and return True if it's reachable, False otherwise."""
    response = ping(host)
    if response is None:
        return False
    else:
        return True

def check_http(url):
    """Send a GET request to the URL and return True if the response status is 200, False otherwise."""
    try:
        response = requests.get(url, timeout=5)  # Adding a timeout for faster response
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Error checking HTTP for {url}: {e}")
        return False

def main():
    minio_address = "103.20.60.90:9001"
    api_address = "http://103.20.60.90:9001"
    server_addresses = [
        "192.168.3.29",
        "192.168.3.30",
        "192.168.3.31",
        "192.168.3.32"
    ]

    # Check Minio server
    minio_host, minio_port = minio_address.split(":")
    print(f"Checking Minio server at {minio_address}...")
    if check_ping(minio_host):
        print("Minio server is reachable.")
    else:
        print("Minio server is not reachable.")

    # Check API server
    print(f"Checking API server at {api_address}...")
    if check_http(api_address):
        print("API server is healthy.")
    else:
        print("API server is not reachable.")

    # Check other servers
    for server_address in server_addresses:
        print(f"Checking server at {server_address}...")
        if check_ping(server_address):
            print(f"Server {server_address} is reachable.")
        else:
            print(f"Server {server_address} is not reachable.")

if __name__ == "__main__":
    main()
