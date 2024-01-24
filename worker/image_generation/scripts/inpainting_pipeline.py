from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from stable_diffusion.model_paths import CLIP_TOKENIZER_DIR_PATH, CLIP_TEXT_MODEL_DIR_PATH, NED_INPAINTING_PATH, DREAMSHAPER_INPAINTING_PATH, INPAINTING_CONFIG_FILE
from utility.labml.monit import section
from utility.utils_logger import logger
from PIL import Image
import torch

class StableDiffusionInpaintingPipeline:
    def __init__(self,
                 model_type="ned",
                 denoising_strength=0.75,
                 guidance_scale=7.5,
                 steps=40,
                 width=512,
                 height=512
                 ):
        # set parameters
        self.denoising_strength=denoising_strength
        self.guidance_scale=guidance_scale
        self.steps=steps
        self.width=width
        self.height=height

        # get model path
        if model_type=="ned":
            self.inpainting_model_path= NED_INPAINTING_PATH
        elif model_type=="dreamshaper":
            self.inpainting_model_path= DREAMSHAPER_INPAINTING_PATH

        # get device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def load_models(self, tokenizer_path=CLIP_TOKENIZER_DIR_PATH, text_encoder_path=CLIP_TEXT_MODEL_DIR_PATH):

        with section("Loading Tokenizer and Text encoder"):
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True, return_tensors="pt", padding=True, truncation=True)
            logger.debug(f"Tokenizer successfully loaded from : {tokenizer_path}")
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, local_files_only=True,
                                                             use_safetensors=True).eval().to(self.device)
            
            self.transformer = self.transformer.to(device=self.device)
            print(self.device)
            logger.debug(f"Text encoder model successfully loaded from : {text_encoder_path}")

        # Load inpainting model
        with section("Loading Inpainting model"):
            self.inpainting_model = StableDiffusionInpaintPipeline.from_single_file(
                pretrained_model_link_or_path=self.inpainting_model_path,
                config_files={'v1': INPAINTING_CONFIG_FILE},
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                local_files_only=True, use_safetensors=True, load_safety_checker=False
            ).to(self.device)
            logger.debug(f"Inpainting model successfully loaded from : {text_encoder_path}")
    
    def inpaint(self, prompt:str, image: Image, image_mask: Image):
        # resizing and converting initial image and mask
        init_image = image.convert("RGB").resize((self.width, self.height))
        mask = image_mask.convert("RGB").resize((self.width, self.height))

        # with torch.no_grad():
        output = self.inpainting_model(
            prompt=prompt, 
            image=init_image, 
            mask_image=mask, 
            num_inference_steps=self.steps, 
            strength=self.denoising_strength, 
            guidance_scale=self.guidance_scale
        )

        return output.images[0]

# # load image & mask
# init_image = Image.open(args.image_path).convert("RGB").resize(args.target_size)
# mask = Image.open(args.mask_path).convert("RGB").resize(args.target_size)

# # Load tokenizer and text encoder
# tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
# text_encoder = CLIPTextModel.from_pretrained(args.text_model_path, local_files_only=True).cuda().eval()

# # load model
# torch_dtype = torch.float32 if args.use_float32 else torch.float16
# pipe = StableDiffusionInpaintPipeline.from_single_file(
#     pretrained_model_link_or_path=args.model_file_path,
#     config_files={'v1': args.config_file_path},
#     text_encoder=text_encoder,
#     tokenizer=tokenizer,
#     local_files_only=True, use_safetensors=True, load_safety_checker=False
# ).to('cuda')

# # do inpainting
# with torch.no_grad():
#     output = pipe(
#         prompt=args.prompt, 
#         image=init_image, 
#         mask_image=mask, 
#         num_inference_steps=args.num_inference_steps, 
#         strength=args.strength, 
#         guidance_scale=args.guidance_scale
#     )

# # save result
# output_image = output.images[0]
# output_image.save(args.target_path)