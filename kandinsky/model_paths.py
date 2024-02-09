from configs.model_config import ModelPathConfig

config = ModelPathConfig(check_existence=False)


class CLIPconfigs:
    TXT_EMB_MODEL = 'kandinsky/kandinsky-2-2-prior/text_encoder'
    TXT_EMB_TOKENIZER = 'kandinsky/kandinsky-2-2-prior/tokenizer'

    IMG_ENC_PROCESSOR = 'kandinsky/kandinsky-2-2-prior/image_encoder'
    IMG_ENC_VISION = 'kandinsky/kandinsky-2-2-prior/image_processor'


class KandinskyConfigs:
    PRIOR_MODEL= "kandinsky/kandinsky-2-2-prior"
    DECODER_MODEL= "kandinsky/kandinsky-2-2-decoder"
    INPAINT_DECODER_MODEL= "kandinsky/kandinsky-2-2-decoder-inpaint"


TXT_EMB_MODEL_PATH = config.get_model_folder_path(CLIPconfigs.TXT_EMB_MODEL)
TOKENIZER_DIR_PATH = config.get_model_folder_path(CLIPconfigs.TXT_EMB_TOKENIZER)

IMAGE_PROCESSOR_DIR_PATH = config.get_model_folder_path(CLIPconfigs.IMG_ENC_PROCESSOR)
VISION_MODEL_DIR_PATH = config.get_model_folder_path(CLIPconfigs.IMG_ENC_VISION)

PRIOR_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.PRIOR_MODEL)
DECODER_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.DECODER_MODEL)
INPAINT_DECODER_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.INPAINT_DECODER_MODEL)
