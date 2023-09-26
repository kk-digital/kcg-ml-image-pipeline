class ImageGenerationTask:

    def __init__(self, generation_task_type, positive_prompt, negative_prompt, model_name, cfg_strength, seed, output_path,
                 num_images, image_width, image_height, batch_size, sampler, steps, output_image_hash):
        self.generation_task_type = generation_task_type
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.model_name = model_name
        self.cfg_strength = cfg_strength
        self.seed = seed
        self.output_path = output_path
        self.num_images = num_images
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.sampler = sampler
        self.steps = steps
        self.output_image_hash = output_image_hash

    def to_dict(self):
        return {
            'generation_task_type': self.generation_task_type,
            'positive_prompt': self.positive_prompt,
            'negative_prompt': self.negative_prompt,
            'model_name': self.model_name,
            'cfg_strength': self.cfg_strength,
            'seed': self.seed,
            'output_path': self.output_path,
            'num_images': self.num_images,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'batch_size': self.batch_size,
            'sampler': self.sampler,
            'steps': self.steps,
            'output_image_hash': self.output_image_hash,
        }

    def from_dict(data):
        return ImageGenerationTask(
            generation_task_type=data.get('generation_task_type', ''),
            positive_prompt=data.get('positive_prompt', ''),
            negative_prompt=data.get('negative_prompt', ''),
            cfg_strength=data.get('cfg_strength', 7),
            model_name=data.get('model_name', ''),
            seed=data.get('seed', ''),
            output_path=data.get('output_path', ''),
            num_images=data.get('num_images', 1),
            image_width=data.get('image_width', 512),
            image_height=data.get('image_height', 512),
            batch_size=data.get('batch_size', 1),
            sampler=data.get('sampler', 'ddim'),
            steps=data.get('steps', 50),
            output_image_hash=data.get('output_image_hash', ''),
        )
