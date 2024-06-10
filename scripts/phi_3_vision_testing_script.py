# from PIL import Image 
# import requests 
# from transformers import AutoModelForCausalLM 
# from transformers import AutoProcessor 

# model_id = "microsoft/Phi-3-vision-128k-instruct" 

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

# messages = [ 
#     {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
#     {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
#     {"role": "user", "content": "Provide insightful questions to spark discussion."} 
# ] 


# url = "https://cdn.vox-cdn.com/thumbor/h_lLWoS4m9CAFJHuiplgNwWhiNI=/0x0:1920x1080/1400x1400/filters:focal(960x540:961x541)/cdn.vox-cdn.com/uploads/chorus_asset/file/24474865/Switch_NSO_GBA_MetroidFusion_March2023_SCRN_01.jpg"
#  #"https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
# image = Image.open(requests.get(url, stream=True).raw) 

# prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

# generation_args = { 
#     "max_new_tokens": 500, 
#     "temperature": 0.0, 
#     "do_sample": False, 
# } 

# generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# # remove input tokens 
# generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

# print(response) 




################ V2
# from PIL import Image
# import requests
# from transformers import AutoModelForCausalLM, AutoProcessor

# model_id = "microsoft/Phi-3-vision-128k-instruct"

# try:
#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, 
#         device_map="cuda", 
#         trust_remote_code=True, 
#         torch_dtype="auto", 
#         _attn_implementation='flash_attention_2'
#     )  # use _attn_implementation='eager' to disable flash attention

#     # Load processor
#     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# except EnvironmentError as e:
#     print(f"Error loading model: {e}")
#     print("Please check your internet connection or verify the model path.")
#     exit(1)

# messages = [
#     {"role": "user", "content": "\nWhat is shown in this image?"},
#     {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
#     {"role": "user", "content": "Provide insightful questions to spark discussion."}
# ]

# url = "https://cdn.vox-cdn.com/thumbor/h_lLWoS4m9CAFJHuiplgNwWhiNI=/0x0:1920x1080/1400x1400/filters:focal(960x540:961x541)/cdn.vox-cdn.com/uploads/chorus_asset/file/24474865/Switch_NSO_GBA_MetroidFusion_March2023_SCRN_01.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

# generation_args = {
#     "max_new_tokens": 500,
#     "temperature": 0.0,
#     "do_sample": False,
# }

# generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# # Remove input tokens
# generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(response)




############### V3

# from PIL import Image
# import requests
# from transformers import AutoModelForCausalLM, AutoProcessor

# # Check internet connectivity
# try:
#     response = requests.get("https://huggingface.co")
#     if response.status_code == 200:
#         print("Internet connection is working.")
#     else:
#         print("Internet connection problem: ", response.status_code)
# except requests.ConnectionError:
#     print("No internet connection.")
#     exit(1)

# # Ensure Pydantic is properly installed
# try:
#     import pydantic
#     print("Pydantic is installed and working.")
# except ImportError as e:
#     print(f"Error importing Pydantic: {e}")
#     print("Please install Pydantic using 'pip install pydantic'")
#     exit(1)

# model_id = "microsoft/Phi-3-vision-128k-instruct"

# try:
#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id, 
#         device_map="cuda", 
#         trust_remote_code=True, 
#         torch_dtype="auto", 
#         _attn_implementation='flash_attention_2'
#     )  # use _attn_implementation='eager' to disable flash attention

#     # Load processor
#     processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# except EnvironmentError as e:
#     print(f"Error loading model: {e}")
#     print("Please check your internet connection or verify the model path.")
#     exit(1)

# messages = [
#     {"role": "user", "content": "\nWhat is shown in this image?"},
#     {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
#     {"role": "user", "content": "Provide insightful questions to spark discussion."}
# ]

# url = "https://cdn.vox-cdn.com/thumbor/h_lLWoS4m9CAFJHuiplgNwWhiNI=/0x0:1920x1080/1400x1400/filters:focal(960x540:961x541)/cdn.vox-cdn.com/uploads/chorus_asset/file/24474865/Switch_NSO_GBA_MetroidFusion_March2023_SCRN_01.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

# generation_args = {
#     "max_new_tokens": 500,
#     "temperature": 0.0,
#     "do_sample": False,
# }

# generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# # Remove input tokens
# generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
# response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(response)




################## V4
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

# Check internet connectivity
try:
    response = requests.get("https://huggingface.co")
    if response.status_code == 200:
        print("Internet connection is working.")
    else:
        print("Internet connection problem: ", response.status_code)
except requests.ConnectionError:
    print("No internet connection.")
    exit(1)

# Ensure Pydantic is properly installed
try:
    import pydantic
    print("Pydantic is installed and working.")
except ImportError as e:
    print(f"Error importing Pydantic: {e}")
    print("Please install Pydantic using 'pip install pydantic'")
    exit(1)

model_id = "microsoft/Phi-3-vision-128k-instruct"

try:
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2'
    )  # use _attn_implementation='eager' to disable flash attention

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

except EnvironmentError as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection or verify the model path.")
    exit(1)

messages = [
    {"role": "user", "content": "\nWhat is shown in this image?"},
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."},
    {"role": "user", "content": "Provide insightful questions to spark discussion."}
]

url = "https://cdn.vox-cdn.com/thumbor/h_lLWoS4m9CAFJHuiplgNwWhiNI=/0x0:1920x1080/1400x1400/filters:focal(960x540:961x541)/cdn.vox-cdn.com/uploads/chorus_asset/file/24474865/Switch_NSO_GBA_MetroidFusion_March2023_SCRN_01.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 500,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# Remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
