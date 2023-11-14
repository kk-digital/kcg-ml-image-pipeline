class ABData:
    def __init__(self, task, username, hash_image_1, hash_image_2, selected_image_index, selected_image_hash,
                 image_archive, image_1_path, image_2_path, datetime, flagged=False):
        self.task = task
        self.username = username
        self.hash_image_1 = hash_image_1
        self.hash_image_2 = hash_image_2
        self.selected_image_index = selected_image_index
        self.selected_image_hash = selected_image_hash
        self.image_archive = image_archive
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
        self.datetime = datetime
        self.flagged = flagged