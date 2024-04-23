import csv
from datetime import datetime
import sys

base_dir = './'
sys.path.insert(0, base_dir)

from utility.http.request import http_get_all_image_hashes_with_global_id

def main():

    image_hashes_with_global_id = http_get_all_image_hashes_with_global_id()
    columns = ["image_hash", "image_global_id"]

    fpath = ("output/{}_image_hashes_with_global_id.csv"
             .format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    try:
        with open(fpath, mode='w', newline='') as file: # write data into fpath
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for data in image_hashes_with_global_id:
                writer.writerow(data)
        print("Successfully saved into !!!", fpath)
    except Exception as e:
        print("Error in saving the image hashes with global id:", e)

if __name__ == "__main__":
    main()