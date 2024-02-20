import msgpack


class GeneratedImg2imgData:
    job_uuid: str
    creation_time: str
    dataset: str
    file_path: str
    file_hash: str
    seed: int
    strength: float
    decoder_steps: int
    decoder_guidance_scale: float
    image_width: int
    image_height: int

    def __init__(self, job_uuid, creation_time, dataset, file_path, file_hash, seed,
                image_width, image_height, strength, decoder_steps, decoder_guidance_scale):
        
        self.job_uuid = job_uuid
        self.creation_time = creation_time
        self.dataset = dataset
        self.file_path = file_path
        self.file_hash = file_hash
        self.seed = seed
        self.image_width = image_width
        self.image_height = image_height
        self.strength= strength
        self.decoder_steps= decoder_steps
        self.decoder_guidance_scale= decoder_guidance_scale

    def serialize(self):
        # Convert object to a dictionary
        return {
            "job_uuid": self.job_uuid,
            "creation_time": self.creation_time,
            "dataset": self.dataset,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "seed": self.seed,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "strength": self.strength,
            "decoder_steps": self.decoder_steps,
            "decoder_guidance_scale": self.decoder_guidance_scale        
        }

    @classmethod
    def deserialize(cls, data):
        # Convert dictionary back to object

        return cls(data["job_uuid"],
                   data["creation_time"],
                   data["dataset"],
                   data["file_path"],
                   data["file_hash"],
                   data["seed"],
                   data["image_width"],
                   data["image_height"],
                   data["strength"],
                   data["decoder_steps"],
                   data["decoder_guidance_scale"])

    def get_msgpack_string(self):
        serialized = self.serialize()
        return msgpack.packb(serialized, use_single_float=True)

    @classmethod
    def from_msgpack_string(cls, msgpack_string):
        data = msgpack.unpackb(msgpack_string, raw=False)
        return cls.deserialize(data)