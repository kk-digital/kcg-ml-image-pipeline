from pymongo import MongoClient

# Replace with your MongoDB connection string
client = MongoClient("mongodb://192.168.3.1:32017/")
db = client["orchestration-job-db"]

def format_size_gb(size_in_bytes):
    size_in_gb = size_in_bytes / (1024 ** 3)
    return f"{size_in_gb:.5f} GB"

def get_collection_sizes():
    collection_sizes = {}
    for collection_name in db.list_collection_names():
        collection_stats = db.command("collstats", collection_name)
        used_size_gb = format_size_gb(collection_stats["size"] + collection_stats["totalIndexSize"])
        num_objects = collection_stats["count"]

        # Calculate min, max, and average object sizes manually
        sizes = [len(str(doc).encode('utf-8')) for doc in db[collection_name].find()]
        if sizes:
            min_obj_size_bytes = min(sizes)
            max_obj_size_bytes = max(sizes)
            avg_obj_size_bytes = sum(sizes) / len(sizes)
        else:
            min_obj_size_bytes = 0
            max_obj_size_bytes = 0
            avg_obj_size_bytes = 0

        collection_sizes[collection_name] = {
            "used_size_gb": used_size_gb,
            "number_of_objects": num_objects,
            "average_object_size_bytes": avg_obj_size_bytes,
            "min_object_size_bytes": min_obj_size_bytes,
            "max_object_size_bytes": max_obj_size_bytes
        }
    return collection_sizes

def save_to_txt(file_path, data):
    with open(file_path, 'w') as file:
        file.write("{\n")
        for collection, stats in data.items():
            file.write(f'    "{collection}": {{\n')
            file.write(f'        "used_size_gb": "{stats["used_size_gb"]}"\n')
            file.write(f'    }},\n')
        file.write("}\n")

if __name__ == "__main__":
    collection_sizes = get_collection_sizes()
    save_to_txt("collection_sizes.txt", collection_sizes)
