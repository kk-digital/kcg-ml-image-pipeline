import msgpack


class GeneratedImageData:
    job_uuid: str
    creation_time: str
    dataset: str
    file_path: str
    file_hash: str
    positive_prompt: str
    negative_prompt: str
    cfg_strength: int
    seed: int
    image_width: int
    image_height: int
    sampler: str
    sampler_steps: int

    def __init__(self, job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt, negative_prompt,
                 cfg_strength, seed, image_width, image_height, sampler, sampler_steps,
                 prompt_scoring_model, prompt_score, prompt_generation_policy, top_k):
        self.job_uuid = job_uuid
        self.creation_time = creation_time
        self.dataset = dataset
        self.file_path = file_path
        self.file_hash = file_hash
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.cfg_strength = cfg_strength
        self.seed = seed
        self.image_width = image_width
        self.image_height = image_height
        self.sampler = sampler
        self.sampler_steps = sampler_steps
        self.prompt_scoring_model = prompt_scoring_model
        self.prompt_score = prompt_score
        self.prompt_generation_policy = prompt_generation_policy
        self.top_k = top_k

    def serialize(self):
        # Convert object to a dictionary
        return {
            "job_uuid": self.job_uuid,
            "creation_time": self.creation_time,
            "dataset": self.dataset,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "cfg_strength": self.cfg_strength,
            "seed": self.seed,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "sampler": self.sampler,
            "sampler_steps": self.sampler_steps,
            "prompt_scoring_model": self.prompt_scoring_model,
            "prompt_score": self.prompt_score,
            "prompt_generation_policy": self.prompt_generation_policy,
            "top_k": self.top_k,
        }

    @classmethod
    def deserialize(cls, data):
        # Convert dictionary back to object
        return cls(data["job_uuid"],
                   data["creation_time"],
                   data["dataset"],
                   data["file_path"],
                   data["file_hash"],
                   data["positive_prompt"],
                   data["negative_prompt"],
                   data["cfg_strength"],
                   data["seed"],
                   data["image_width"],
                   data["image_height"],
                   data["sampler"],
                   data["sampler_steps"],
                   data["prompt_scoring_model"],
                   data["prompt_score"],
                   data["prompt_generation_policy"],
                   data["top_k"])

    def get_msgpack_string(self):
        serialized = self.serialize()
        return msgpack.packb(serialized)

    @classmethod
    def from_msgpack_string(cls, msgpack_string):
        data = msgpack.unpackb(msgpack_string.encode('latin1'), raw=False)
        return cls.deserialize(data)