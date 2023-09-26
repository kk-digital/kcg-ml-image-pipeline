class IconGenerationTask:

    def __init__(self, generation_task_type, prompt, model_name, cfg_strength, seed, output_path,
                 num_images, image_width, image_height, batch_size, checkpoint_path, device, sampler, steps,
                 prompt_list_dataset_path, init_img, init_mask, output_image_hash, mask_blur, inpainting_fill, styles,
                 resize_mode, denoising_strength, image_cfg_scale, inpaint_full_res_padding, inpainting_mask_invert):
        self.generation_task_type = generation_task_type
        self.prompt = prompt
        self.model_name = model_name
        self.cfg_strength = cfg_strength
        self.seed = seed
        self.output_path = output_path
        self.num_images = num_images
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.sampler = sampler
        self.steps = steps
        self.prompt_list_dataset_path = prompt_list_dataset_path
        self.init_img = init_img
        self.init_mask = init_mask
        self.output_image_hash = output_image_hash
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.styles = styles
        self.resize_mode = resize_mode
        self.denoising_strength = denoising_strength
        self.image_cfg_scale = image_cfg_scale
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert

    def to_dict(self):
        return {
            'generation_task_type': self.generation_task_type,
            'prompt': self.prompt,
            'model_name': self.model_name,
            'cfg_strength': self.cfg_strength,
            'seed': self.seed,
            'output_path': self.output_path,
            'num_images': self.num_images,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'batch_size': self.batch_size,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'sampler': self.sampler,
            'steps': self.steps,
            'prompt_list_dataset_path': self.prompt_list_dataset_path,
            'init_img': self.init_img,
            'init_mask': self.init_mask,
            'output_image_hash': self.output_image_hash,

            'mask_blur': self.mask_blur,
            'inpainting_fill': self.inpainting_fill,
            'styles': self.styles,
            'resize_mode': self.resize_mode,
            'denoising_strength': self.denoising_strength,
            'image_cfg_scale': self.image_cfg_scale,
            'inpaint_full_res_padding': self.inpaint_full_res_padding,
            'inpainting_mask_invert': self.inpainting_mask_invert,
        }

    def from_dict(data):
        return IconGenerationTask(
            generation_task_type=data.get('generation_task_type', ''),
            prompt=data.get('prompt', ''),
            cfg_strength=data.get('cfg_strength', 7),
            model_name=data.get('model_name', ''),
            seed=data.get('seed', ''),
            output_path=data.get('output_path', ''),
            num_images=data.get('num_images', 1),
            image_width=data.get('image_width', 512),
            image_height=data.get('image_height', 512),
            batch_size=data.get('batch_size', 1),
            checkpoint_path=data.get('checkpoint_path', ''),
            device=data.get('device', 'cuda'),
            sampler=data.get('sampler', 'ddim'),
            steps=data.get('steps', 50),
            prompt_list_dataset_path=data.get('prompt_list_dataset_path', ''),
            init_img=data.get('init_img', ''),
            init_mask=data.get('init_mask', ''),
            output_image_hash=data.get('output_image_hash', ''),

            mask_blur=data.get('mask_blur', 4),
            inpainting_fill=data.get('inpainting_fill', 1),
            styles=data.get('styles', []),
            resize_mode=data.get('resize_mode', 0),
            denoising_strength=data.get('denoising_strength', 0.75),
            image_cfg_scale=data.get('image_cfg_scale', 1.5),
            inpaint_full_res_padding=data.get('inpaint_full_res_padding', 32),
            inpainting_mask_invert=data.get('inpainting_mask_invert', 0),
        )
