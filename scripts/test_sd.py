# #################
#  在modelscope 的GPU notebook环境经行部署设置
# 
# 
# 
# 
# 

# from openai import OpenAI

# model_id = 'tensorart/stable-diffusion-3.5-medium-turbo'

# client = OpenAI(
#     base_url='https://ms-fc-de5ab47a-a2b4.api-inference.modelscope.cn/v1',
#     api_key='e37bfdad-0f6a-46c2-a7bf-f9dc365967e3'
# )

# response=client.chat.completions.create(
#     model=model_id,
#     messages=[{"role":"user", "content":"画一只猫的图片？"}],
#     stream=True
# )

# for chunk in response:
#     print(chunk.choices[0].delta.content, end='', flush=True)



# import torch
# from diffusers import StableDiffusion3Pipeline
# import numpy as np
# from safetensors.torch import load_file
# from huggingface_hub import hf_hub_download

# repo = "tensorart/stable-diffusion-3.5-medium-turbo"
# ckpt = "lora_sd3.5m_turbo_8steps.safetensors"

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16,)
                                                
# pipe = pipe.to("cuda")

# pipe.load_lora_weights(hf_hub_download(repo, ckpt))
# pipe.fuse_lora()


# pipe = pipe.to("cuda")

# image = pipe(
#    "A beautiful bald girl with silver and white futuristic metal face jewelry, her full body made of intricately carved liquid glass in the style of Tadashi, the complexity master of cyberpunk, in the style of James Jean and Peter Mohrbacher. This concept design is trending on Artstation, with sharp focus, studio-quality photography, and highly detailed, intricate details.",
#    num_inference_steps=8,
#    guidance_scale=1.5,
#    height=1024,
#    width=768 
# ).images[0]
# image.save("./test1.webp")




# import gradio as gr

# with gr.Blocks(fill_height=True) as demo:
#     with gr.Row():
#         gr.Markdown("# Inference Provider")
#         gr.Markdown("This Space showcases the black-forest-labs/FLUX.1-dev model, served by the fal-ai API. Sign in with your Hugging Face account to use this API.")
#         button = gr.LoginButton("Sign in")
#     gr.load("models/black-forest-labs/FLUX.1-dev", accept_token=button, provider="fal-ai")
    
# demo.launch()

# import gradio as gr
# import os
# os.environ["HF_TOKEN"] = "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"

# gr.load(
#    "models/black-forest-labs/FLUX.1-schnell",

#    hf_token= "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"
# ).launch()



#  testing public FlUX model to generate image 
# 
# 
# 
# 
# 
# 

# from huggingface_hub import InferenceClient

# client = InferenceClient(
# 	token="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"
# )

# output is a PIL.Image object
# import random
# for i in range(5):
# 	seed=random.randint(0,100)
# 	image = client.text_to_image(
# 		f"given a random seed: {seed}, generate image with Prompts: Astronaut riding a horse ",
# 		model="black-forest-labs/FLUX.1-schnell"
# 	)
# 	image.save(f"./new_image_{i}.png")


from rembg import remove , new_session
from PIL import Image
input_image = Image.open( '../examples/results/generate a women dress with summer style_0.png')
session = new_session(model_name='u2net')
output_image = remove(input_image, session=session)
output_image.save("./tmp_v2.png")

# import torch
# from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
# image = pipe(
#     prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-dev.png")
