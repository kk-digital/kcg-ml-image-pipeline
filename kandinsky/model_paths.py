from configs.model_config import ModelPathConfig

config = ModelPathConfig(check_existence=False)


class KandinskyConfigs:
    PRIOR_MODEL= "kandinsky/kandinsky-2-2-prior"
    DECODER_MODEL= "kandinsky/kandinsky-2-2-decoder"
    INPAINT_DECODER_MODEL= "kandinsky/kandinsky-2-2-decoder-inpaint"

PRIOR_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.PRIOR_MODEL)
DECODER_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.DECODER_MODEL)
INPAINT_DECODER_MODEL_PATH = config.get_model_folder_path(KandinskyConfigs.INPAINT_DECODER_MODEL)
