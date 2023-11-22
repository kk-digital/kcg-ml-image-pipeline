class ABData:
    def __init__(self, task, username, hash_image_1, hash_image_2, selected_image_index, selected_image_hash,
                 image_1_path, image_2_path, datetime, flagged=False):
        self.task = task
        self.username = username
        self.hash_image_1 = hash_image_1
        self.hash_image_2 = hash_image_2
        self.selected_image_index = selected_image_index
        self.selected_image_hash = selected_image_hash
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
        self.datetime = datetime
        self.flagged = flagged

    @classmethod
    def deserialize(cls, data: dict):
        # Convert dictionary back to object

        flagged = False
        if "flagged" in data:
            flagged = data["flagged"]

        return cls(task=data["task"],
                   username=data["username"],
                   hash_image_1=data["image_1_metadata"]["file_hash"],
                   hash_image_2=data["image_2_metadata"]["file_hash"],
                   selected_image_index=data["selected_image_index"],
                   selected_image_hash=data["selected_image_hash"],
                   image_1_path=data["image_1_metadata"]["file_path"],
                   image_2_path=data["image_2_metadata"]["file_path"],
                   datetime=data["datetime"],
                   flagged=flagged)
