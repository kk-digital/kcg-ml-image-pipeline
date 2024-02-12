import msgpack


class GeneratedImageData:
    job_uuid: str
    creation_time: str
    dataset: str
    file_path: str
    file_hash: str
    positive_prompt: str
    negative_prior_prompt: str
    negative_encoder_prompt: str
    strength: float
    prior_steps: int
    decoder_steps: int
    prior_guidance_scale: float
    decoder_guidance_scale: float
    image_width: int
    image_height: int
    prompt_scoring_model: str
    prompt_score: float
    prompt_generation_policy: str
    top_k: float

    def __init__(self, job_uuid, creation_time, dataset, file_path, file_hash, positive_prompt,
                    negative_prior_prompt, negative_encoder_prompt,
                    image_width, image_height, strength, decoder_steps, prior_steps,
                    prior_guidance_scale, decoder_guidance_scale, prompt_scoring_model,
                    prompt_score, prompt_generation_policy, top_k):
        
        self.job_uuid = job_uuid
        self.creation_time = creation_time
        self.dataset = dataset
        self.file_path = file_path
        self.file_hash = file_hash
        self.positive_prompt = positive_prompt
        self.negative_prior_prompt = negative_prior_prompt
        self.negative_encoder_prompt = negative_encoder_prompt
        self.image_width = image_width
        self.image_height = image_height
        self.strength= strength
        self.prior_steps= prior_steps
        self.decoder_steps= decoder_steps
        self.prior_guidance_scale= prior_guidance_scale
        self.decoder_guidance_scale= decoder_guidance_scale
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
            "negative_prior_prompt": self.negative_prior_prompt,
            "negative_encoder_prompt": self.negative_encoder_prompt,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "strength": self.strength,
            "prior_steps": self.prior_steps,
            "decoder_steps": self.decoder_steps,
            "prior_guidance_scale": self.prior_guidance_scale,
            "decoder_guidance_scale": self.decoder_guidance_scale,
            "prompt_scoring_model": self.prompt_scoring_model,
            "prompt_score": self.prompt_score,
            "prompt_generation_policy": self.prompt_generation_policy,
            "top_k": self.top_k,
        }

    @classmethod
    def deserialize(cls, data):
        # Convert dictionary back to object
        prompt_scoring_model = None
        if "prompt_scoring_model" in data:
            prompt_scoring_model = data["prompt_scoring_model"]

        prompt_score = None
        if "prompt_score" in data:
            prompt_score = data["prompt_score"]

        prompt_generation_policy = None
        if "prompt_generation_policy" in data:
            prompt_generation_policy = data["prompt_generation_policy"]

        top_k = None
        if "top_k" in data:
            top_k = data["top_k"]

        return cls(data["job_uuid"],
                   data["creation_time"],
                   data["dataset"],
                   data["file_path"],
                   data["file_hash"],
                   data["positive_prompt"],
                   data["negative_prior_prompt"],
                   data["negative_encoder_prompt"],
                   data["image_width"],
                   data["image_height"],
                   data["strength"],
                   data["prior_steps"],
                   data["decoder_steps"],
                   data["prior_guidance_scale"],
                   data["decoder_guidance_scale"],
                   prompt_scoring_model,
                   prompt_score,
                   prompt_generation_policy,
                   top_k)

    def get_msgpack_string(self):
        serialized = self.serialize()
        return msgpack.packb(serialized, use_single_float=True)

    @classmethod
    def from_msgpack_string(cls, msgpack_string):
        data = msgpack.unpackb(msgpack_string, raw=False)
        return cls.deserialize(data)