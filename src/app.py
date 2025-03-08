# Dependencies: gradio, fire, langchain, openai, numpy, ffmpeg, moviepy
# API Reference: https://www.gradio.app/docs/,
# https://github.com/zhayujie/chatgpt-on-wechat, https://docs.link-ai.tech/platform/api,  https://docs.link-ai.tech/api#/
# Description: This file contains the code to run the gradio app for the movie generator.
# 
#
#
# 参考链接： https://zhuanlan.zhihu.com/p/684798694
#
#
####################################################################################################

import gradio as gr
import fire
from gradio_client import Client, file
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

from openai import OpenAI
import os
# import moviepy.editor as mppyth
# from moviepy.editor import *
# from movie_generator.agi.suno.suno import Suno
import requests


import ollama
from ollama import chat
from ollama import ChatResponse
import random
import os
import requests
import json
# image package
from PIL import Image
from rembg import remove 
from PIL import Image
from io import BytesIO


os.environ['HF_TOKEN']="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"
os.environ["STABILITY_KEY"]="sk-RHtzI49J5ke7u1NjaoR05cmvBYKEzHWIf9xRizu5oZ0ylu18"
from huggingface_hub import InferenceClient
class MyAgent:
    def __init__(self,hf_token):
        self.hf_token=hf_token
        self.client = InferenceClient(
            token=self.hf_token
        )

        #@title Connect to the Stability API

        import getpass
        # @markdown To get your API key visit https://platform.stability.ai/account/keys
        # STABILITY_KEY = getpass.getpass('Enter your API Key')

        pass

    #@title Define functions

    def send_generation_request(self, host,params):
        k = os.environ["STABILITY_KEY"]
        headers = {
            "Accept": "image/*",
            "Authorization": f"Bearer {k}"
        }

        # Encode parameters
        files = {}
        image = params.pop("image", None)
        mask = params.pop("mask", None)
        if image is not None and image != '':
            files["image"] = open(image, 'rb')
        if mask is not None and mask != '':
            files["mask"] = open(mask, 'rb')
        if len(files)==0:
            files["none"] = ''

        # Send request
        print(f"Sending REST request to {host}...")
        response = requests.post(
            host,
            headers=headers,
            files=files,
            data=params
        )
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        return response


    def image2image(self, 
                    result_path = './examples/results/',
                    image = "/generate a sexy red women dress_0.png",
                    prompt = "a girl is laugthing",
                    negative_prompt = "",
                    seed = 0,
                    output_format = "jpeg",
                    strength = 0.75
        ):
        #@title SD3.5 Large
        #@markdown - Drag and drop image to file folder on left
        #@markdown - Right click it and choose Copy path
        #@markdown - Paste that path into image field below
        #@markdown <br><br>

        host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

        params = {
            "image" : image,
            "prompt" : prompt,
            "negative_prompt" : negative_prompt,
            "strength" : strength,
            "seed" : seed,
            "output_format": output_format,
            "model" : "sd3.5-large",
            "mode" : "image-to-image"
        }

        response = self.send_generation_request(
            host,
            params
        )

        # Decode response
        output_image = response.content
        finish_reason = response.headers.get("finish-reason")
        seed = response.headers.get("seed")

        # Check for NSFW classification
        if finish_reason == 'CONTENT_FILTERED':
            raise Warning("Generation failed NSFW classifier")

        # Save and display result
        generated =os.path.join(result_path, f"generated_{seed}.{output_format}") 
        with open(generated, "wb") as f:
            f.write(output_image)
        print(f"Saved image {generated}")
        print("Result image:")
        return generated



    def call_LLM(self, inputs, prompts= '你是一个时尚服装行业的专家， 请回答下面问题：', model_version = 'Qwen'):
        inputs = prompts + ' ' + inputs
        if model_version=="Qwen":
            from openai import OpenAI

            model_id = 'Qwen/Qwen2.5-3B-Instruct-GGUF'

            client = OpenAI(
                base_url='https://ms-fc-2ea3820b-8c19.api-inference.modelscope.cn/v1',
                api_key='e37bfdad-0f6a-46c2-a7bf-f9dc365967e3'
            )

            response=client.chat.completions.create(
                model=model_id,
                messages=[{"role":"user", "content":inputs}],
                stream=True
            )

            res= []
            for chunk in response:
                # print(chunk.choices[0].delta.content, end='', flush=True)
                res.append(chunk.choices[0].delta.content)
            return "".join(res)
        elif model_version in ['deepseek-r1:1.5b', 'llama3.2:latest']: 
            # model= 'deepseek-r1:1.5b'
            # model = 'llama3.2:latest'
            response: ChatResponse = chat(model= model_version, messages=[
            {
                'role': 'user',
                'content': inputs,
            },
            ])
            return response['message']['content']
        else:
            return "LLM version is not supported yet."
        
    def text_to_image(self, prompt="Astronaut riding a horse",
                      model_version="black-forest-labs/FLUX.1-schnell",
                      negative_prompt=None,
                      width=None,
                      height=None,
                      num_inference_steps=None,
                      guidance_scale=None):
        # output is a PIL.Image object
        if model_version != 'modelscope':
            image = self.client.text_to_image(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                model=model_version
            )
        else:
            url = 'https://api-inference.modelscope.cn/v1/images/generations'

            payload = {
                'model': 'MusePublic/489_ckpt_FLUX_1',#ModelScope Model-Id,required
                'prompt': prompt,
                'height':height,
                'width': width,
                "num_inference_steps":num_inference_steps,
                "guidance_scale":guidance_scale,
            }
            headers = {
                'Authorization': 'Bearer e37bfdad-0f6a-46c2-a7bf-f9dc365967e3',
                'Content-Type': 'application/json'
            }

            response = requests.post(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), headers=headers)

            response_data = response.json()
            image = Image.open(BytesIO(requests.get(response_data['images'][0]['url']).content))
        return image
    

    def cut_image(self, input_img):
        if isinstance(input_img, str):
            input_image = Image.open(input_img)
        else:
            uint8_arr = input_img.astype(np.uint8)
            # 生成 PIL 图像
            input_image = Image.fromarray(uint8_arr)
        output_image = remove(input_image)
        return output_image
        
    def image_to_image(self, image,
                       prompt =None,
                       negative_prompt = None,
                       height = None,
                       width = None,
                       num_inference_steps = None,
                       guidance_scale = None,
                       model_version = "black-forest-labs/FLUX.1-schnell"):
        # Coming Soon
        self.client.image_to_image(image,
                       prompt,
                       negative_prompt,
                       height,
                       width,
                       num_inference_steps,
                       guidance_scale,
                       model = model_version)
        


import os
import subprocess
import pandas as pd
class GradioApp:
    def __init__(self,config=None):
        #config with info of 
        # model version
        # prompts
        #others
        self.agent = MyAgent(os.environ['HF_TOKEN'])
        self.config=config
        MyAgent
        self.config=config
        # self.image_dir = "/mnt/d/workspace/projects/Project_TextImage_Generator/examples"
        self.result_dir = "./examples"
        self.init_folder()
        self.init_variable()
        pass
    def init_folder(self, ):
        self.image_result = os.path.join(self.result_dir, "results") 
        if not os.path.exists(self.image_result):
            os.mkdir(self.image_result)
        else:
            # os.removedirs(self.image_root)
            # subprocess.getstatusoutput(f"rm -rf {self.image_result}")
            # os.mkdir(self.image_result)
            print(f"{self.image_result} already exists")

        
        self.text_result = os.path.join(self.result_dir, "text") 
        if not os.path.exists(self.text_result):
            os.mkdir(self.text_result)
        else:
            # os.removedirs(self.image_root)
            # subprocess.getstatusoutput(f"rm -rf {self.text_result}")
            # os.mkdir(self.text_result)
            print(f"{self.text_result} already exists")
            
        self.model_dir = os.path.join(self.result_dir, "models")
        self.clothes_dir = os.path.join(self.result_dir, "clothes")
        self.reference_dir = os.path.join(self.result_dir, "references")
        self.model_files = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
        self.clothes_files = [os.path.join(self.clothes_dir, f) for f in os.listdir(self.clothes_dir)]
        self.reference_files = [os.path.join(self.reference_dir, f) for f in os.listdir(self.reference_dir)]
    
    def init_variable(self,):
        # initialize list of sentences
        self.text_result_file = os.path.join(self.text_result, "selected_sentences.csv") 
        history_words = []
        if os.path.exists(self.text_result_file):
            df= pd.read_csv(self.text_result_file, header=0)
            history_words = list(df['result'].values)
        self.current_options = [] + history_words
        # 
        ls= [os.path.join(self.image_result,f) for f in os.listdir( self.image_result) if f.endswith(".png") or f.endswith(".jpg")] 
        
        self.current_image_ls_1 = []+ls
        self.current_image_ls_2 = []+ls
        print("data: ", ls, "self.current_image_ls_1: ", self.current_image_ls_1, "self.current_image_ls_2:",self.current_image_ls_2)

    def test_image_func(self, input_image, filter_mode='sepia'):
        def filter_image(input_image, filter_mode='sepia'):
            def sepia(input_img):
                sepia_filter = np.array([
                    [0.393, 0.769, 0.189], 
                    [0.349, 0.686, 0.168], 
                    [0.272, 0.534, 0.131]
                ])
                sepia_img = input_img.dot(sepia_filter.T)
                sepia_img /= sepia_img.max()
                return sepia_img
            def grayscale(input_img):
                input_img = np.mean(input_img, axis=2) / np.max(input_img)
                return input_img
            if filter_mode == 'sepia':
                return sepia(input_image)
            elif filter_mode == 'grayscale':
                return grayscale(input_image)
            else:
                return input_image
        res = f"Got image from image input: {input_image}"
        filtered_image = filter_image(input_image, filter_mode)
        return res, filtered_image
    
    def dress_up_func(self, model_images, cloths_images, prompts, similarity):
        # 请求GPT response
        return "dress_up_func output",[(model_images, "模特"), (cloths_images, "衣服")]*5

    def update_model_func(self, model_images, cloths_images, prompts, similarity):
        # 请求GPT response
        return "update_model_func output", [(model_images, "模特"), (cloths_images, "衣服")]*5
    
    def image_module(self, mode='dress_up', title='image_module', desc=''):
        if mode == 'dress_up':
            # 模特试衣
            func = self.dress_up_func
        elif mode == 'update_model':
            # 更新模特
            func = self.update_model_func
        else:
            func = self.dress_up_func
        examples = []
        for i, (c, m) in enumerate( zip(self.clothes_files, self.model_files) ):
            examples.append([c, m, 'sepia', 0.6] )
        comp = gr.Interface(
                fn= func,
                inputs=[gr.Image(label='衣服', scale=1, height=300),
                        gr.Image(label='模特',scale=1, height=300),
                        gr.Dropdown(['sepia', 'grayscale']),
                        gr.Slider(0, 10, value=5, label="相似度控制", info="similarity between 2 and 20")],
                outputs=[gr.Textbox(label="文本输出"),
                         gr.Gallery(label='图片展示',height='auto',columns=3)
                         ],
                title=title,
                description=desc,
                # theme="huggingface",
                examples=examples,
            )
        return comp
    
    def image_module_v2(self, mode='dress_up', title='image_module', desc=''):
        def upload_file(files, current_files):
            file_paths = current_files + [file.name for file in files]
            return file_paths

        def gen_images(clothes_img, model_img):
            new_images = []
            #call LLM/SD here
            new_images.append(clothes_img)
            new_images.append(model_img)
            return new_images
        
        def clear_images():
            return []
        def slider_func(val):
            print("slider value: ", val)


        if mode == 'dress_up':
            # 模特试衣
            func = self.dress_up_func
        elif mode == 'update_model':
            # 更新模特
            func = self.update_model_func
        else:
            func = self.dress_up_func

        with gr.Blocks() as demo:
            # first row
            with gr.Row():
                # first col -> input column
                with gr.Column():
                    model_image=gr.Image(label="模特图片",type='pil', height=None, width=None)
                    clothes_image=gr.Image(label="衣服图片",type='pil', height=None, width=None)
                    upload_button = gr.UploadButton("选择图片上传 (Upload Photos)", file_types=["image"], file_count="multiple")
                    generate_img_button = gr.Button("生成图片")
                    slider = gr.Slider(0, 10, value=5, label="相似度控制", info="similarity between 2 and 20")
                    clear_button = gr.Button("清空图片 (Clear Photos)")
                    
                    # analyze_button = gr.Button("显示图片信息 (Show Image Info)")
                    input_image_gallery = gr.Gallery(type='pil', label='输入图片列表 (Photos)', height=250, columns=4, visible=True)
                # second col-> output column
                with gr.Column():
                    image_gallery = gr.Gallery(type='pil', label='图片列表 (Photos)', height=250, columns=4, visible=True)
            # user_images = gr.State([])
            # upload_button.upload(upload_file, inputs=[upload_button, user_images], outputs=image_gallery)
            slider.input(fn=slider_func)
            generate_img_button.click(gen_images,inputs=[clothes_image, model_image], outputs= image_gallery)
            clear_button.click(fn=clear_images, inputs=None, outputs=image_gallery)
            # analyze_button.click(get_image_info, inputs=image_gallery, outputs=analysis_output)
            return demo

    def gen_text(self,inputs, LLM_version='Qwen'):
        # 设置前置prompt做限制
        prompts = "你是一个时尚服装行业的专家， 请回答下面问题,只罗列答案不要返回多余的词："
        # model= 'deepseek-r1:1.5b'
        # return call_LLM(inputs,prompts, model_version='llama3.2:latest')
        return self.agent.call_LLM(inputs,prompts, model_version=LLM_version)

    def gen_text_v2(self,inputs, role, task, model_version='Qwen', option_states=[]):
        # 设置前置prompt做限制
        prompts = role + task
        # role "你是一个时尚服装行业的专家"
        #task "请回答下面问题,只罗列答案不要返回多余的词,并以列表形式："
        # model= 'deepseek-r1:1.5b'
        # return call_LLM(inputs,prompts, model_version='llama3.2:latest')
        words = self.agent.call_LLM(inputs,prompts, model_version=model_version)
        updated_options = [t.split(". ")[1] for t in words.split("\n") if ". " in t] + option_states
        return updated_options, gr.update(choices=updated_options)
    
    def gen_text2Image(self, hints, inputs, mode_dropdown, model_version, num_images, num_inference_steps, current_states):
        images = []

        # valid_version = ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]
        num_images = int(num_images)
        print("get num_images: ", num_images)
        print("hints: ", hints)
        if mode_dropdown == '合并生成':
            hints = [ ",".join(hints) ]
        msg_ls = []
        for h in hints:
            prompts = f" Please generate images with following requirements:"
            prompts += "1.prompt: " + inputs +"\n"
            prompts += f"2. hints related to results: {h} \n"
            msg_ls.append(prompts)
            for i in range(num_images):
                seed = random.randint(0, 100)
                prompts += f"3. A random seed: {seed+i*10} of this image."
                # if model_version in valid_version:
                image = self.agent.text_to_image(prompt=prompts, width=8*3*10,height=8*4*10, num_inference_steps=num_inference_steps, model_version=model_version)
                # image = self.agent.client.text_to_image(
                #     prompt=prompts,
                #     model="black-forest-labs/FLUX.1-schnell"
                # )


                image_name = inputs+f"_{seed+i*100}.png"
                print(f"Generated image: {image_name} ")
                # images.append(image)
                image_path = os.path.join(self.image_result, image_name)
                print("Saving file: ", image_path)
                image.save(image_path)
                # images.append((image, image_name))
                images.append((image_path, image_name))
                current_states.append((image_path, image_name))
                # yield images, current_states, "\n".join(msg_ls)
        # print()
        # current_states.extend(images)
        return images, current_states, current_states, "\n".join(msg_ls)


    # def text_module(self, title='文本生成', desc="AI生成关键词"):
    #     comp = gr.Interface(
    #             fn= self.gen_text,
    #             inputs=[gr.Textbox(label="文本输入"), gr.Dropdown(['deepseek-r1:1.5b', 'llama3.2:latest','Qwen'], label='模型选择')],
    #             # outputs=[gr.Textbox(label="结果输出")],
    #             outputs = [gr.CheckboxGroup(label="请选择选项",interactive=False)],
    #             title=title,
    #             description=desc,
    #             theme="huggingface",
    #             examples=[
    #                 ["列出2024年最受欢迎的10个衣服品牌","llama3.2:latest"],
    #                   ["哪些款式的女装比较潮流， 请列出10个女装品类","Qwen"],
    #                   ["随机生成10个衣服类目并列出来","Qwen"]],
    #             cache_examples=True,
    #         )
    #     return comp
    

    def text_module_v2(self, title='AI生词模块', desc="AI生成关键词"):
        import csv
        import os
        # LLM生成词选项状态列表
        

        def add_new_option(new_option, existing_options):
            if new_option.strip() and new_option not in existing_options:
                updated_options = existing_options + [new_option.strip()]
                # 关键修改：使用 gr.update() 更新组件属性
                return updated_options, gr.update(choices=updated_options)
            return existing_options, gr.update()  # 无变化时返回空更新

        def delete_datafile():
            subprocess.getstatusoutput(f"rm -rf {self.text_result_file}")
            return f"Removed datafile: {self.text_result_file} "

        def save_selected(selected_items, options_state):
            if selected_items:
                result_text = "\n".join(selected_items)
                with open(self.text_result_file, "w+", encoding="utf-8") as f:
                    # json.dump(selected_items, f, ensure_ascii=False)
                    writer = csv.writer(f)
                    writer.writerow(["result"])
                    for r in selected_items:
                        writer.writerow([r])
                msg=f"✅ 已保存 {len(selected_items)} 项到 {self.text_result_file}\n 生成结果: \n{result_text}"
                return msg, gr.update(choices=selected_items) 
            return "⚠️ 请先选择需要保存的项目",gr.update(choices=selected_items) 
        
        def gallery_select(evt: gr.SelectData, selected_indices):
            """处理图片选择/取消选择"""
            print("evt: ", evt)
            print("selected_indices: ", selected_indices)
            index = evt.index
            # current = selected_indices.value.copy()
            current = selected_indices[:]
            if not current:
                current.append(index)
            else:
                current[-1] = index
            # if index in current:
            #     current.remove(index)
            # else:
            #     current.append(index)
            
            return current
        

        def process_selected_images(prompt, selected_indices_1, current_images_1,selected_indices_2, current_images_2 ):
                """
                处理选中的图片
                :param selected_indices: 选中图片的索引列表
                :param current_images: 当前图库中的所有图片
                :return: 处理结果（示例为拼接后的图片）
                """
                if not selected_indices_1:
                    return None, "请先选择至少一张图片"
                
                # 获取选中的图片
                selected_1 = [current_images_1[i] for i in selected_indices_1]
                selected_2 = [current_images_2[i] for i in selected_indices_2]
                
                # call LLM to generate image
                result =  [(selected_1[-1], 'result')]
                msg= f"""
                已处理 {len(selected_1)} {len(selected_2)} 张图片\n\n Prompt: {prompt}
                """
                try:
                    input_image =  "/generate a sexy red women dress_0.png" if not selected_1[-1] else selected_1[-1]
                    print("input_image: ", input_image)
                    gen_image = self.agent.image2image( 
                        result_path = self.image_result,
                        image = input_image,
                        prompt = prompt,
                        negative_prompt = "",
                        seed =  random.randint(0, 100),
                        output_format = "jpeg",
                        strength = 0.75)
                    result = [(gen_image, 'result')]
                except Exception as e:
                    print("Error: ", e)
                    msg = "Error: " + str(e)
                    result = [(selected_1[-1], 'result')]

                
                return result, msg

        
        with gr.Blocks() as demo:
            gr.Markdown(f"## Step1: {title}")
            # 状态存储
            options_state = gr.State(value=self.current_options)
            with gr.Row():
                
                # 添加新选项区域
                with gr.Column():
                    # role "你是一个时尚服装行业的专家"
        #task "请回答下面问题,只罗列答案不要返回多余的词,并以列表形式："
                    # new_option_input = gr.Textbox(label="输入Prompts", placeholder="输入要添加的内容...")
                    role_input = gr.Textbox(value="你是一个时尚服装行业的专家,请回答下面问题",label="输入角色", placeholder="输入要添加的内容...")
                    task_input = gr.Textbox(value= "答案要求:只罗列答案不要返回多余的词,并以列表形式返回, 每条列表格式如下:<序号>.<答案>", label="输入任务", placeholder="输入要添加的内容...")
                    user_input = gr.Textbox(value= "列出2024年最受欢迎的10个衣服品牌", label="输入任务", placeholder="输入要添加的内容...")
                    model_options = gr.Dropdown(value='Qwen',  choices=['deepseek-r1:1.5b', 'llama3.2:latest','Qwen'], label='模型选择')
                    add_btn = gr.Button("➕ 生成文本", variant="primary")
                    clear_btn = gr.Button("重置数据文件", variant="primary")
                
                # 选项操作区
                with gr.Column():
                    dynamic_checkbox = gr.CheckboxGroup(
                        label="可选项目",
                        choices=self.current_options,
                        interactive=True,
                        elem_classes=["vertical-checkbox"]
                    )
                    # t2t_select_btn = 
                    save_btn = gr.Button("💾 保存选中项", variant="secondary")
                    output_msg = gr.Markdown()
            

            # 文生图模块
            gr.Markdown(f"")
            gr.Markdown(f"")
            gr.Markdown(f"## Step2: FLUX 文生图")
            
            
            selected_indices_0 = gr.State(value=[])
            selected_indices_1 = gr.State(value=[])
            selected_indices_2 = gr.State(value=[])
            # current_image_states_1 = gr.State(value=self.current_image_ls_1)
            # current_image_states_2 = gr.State(value=self.current_image_ls_2)
            current_image_states = gr.State(value=self.current_image_ls_1)
            current_image_states_1 = gr.State(value=self.current_image_ls_1)
            current_image_states_2 = gr.State(value=self.current_image_ls_2)
            # current_image_states_1= current_image_states_2 = current_image_states
            with gr.Row():
                with gr.Column():
                    t2i_checkbox = gr.CheckboxGroup(
                        label="可选项目",
                        choices=self.current_options,
                        interactive=True,
                        elem_classes=["vertical-checkbox"]
                    )
                    t2i_prompt_input = gr.Textbox(value="generate a sexy red women dress",label="输入prompt")
                    t2i_model_dropdown= gr.Dropdown(value="modelscope",choices= ["black-forest-labs/FLUX.1-schnell","black-forest-labs/FLUX.1-dev", "modelscope"], label='模型选择')
                    t2i_image_num_input = gr.Textbox(value=2,label="生成图片数量")
                    t2i_infer_steps = gr.Slider(0, 10, value=1, label="推理步数", info="similarity between 1 and 10")
                    # t2i_update_btn = gr.Button("加载词", variant="primary")
                    mode_dropdown = gr.Dropdown(value='合并生成',  choices=['合并生成', '逐条生成'], label='模型选择')
                    image_btn = gr.Button("图片生成", variant="primary")
                with gr.Column():
                    t2i_gallery = gr.Gallery(label='图片展示',height='auto',columns=3)
                    
                    t2i_output_msg = gr.Markdown()

            gr.Markdown(f"")
            gr.Markdown(f"")
            gr.Markdown(f"## Step3: 图编辑")
            with gr.Row():
                with gr.Column():
                    im = gr.ImageEditor(crop_size="3:4")
                       
                with gr.Column():
                    im_preview = gr.Image(height=300)
                    cut_bg_btn = gr.Button("抠图") 
                    im_name_text = gr.Textbox(value="new_image_1.png",label="图片名称")
                    save_im_btn = gr.Button("保存图片")
            
            # 文生图模块
            gr.Markdown(f"")
            gr.Markdown(f"")
            gr.Markdown(f"## Step4: 图生图")
            with gr.Row():
                with gr.Column():
                    i2i_prompt_input = gr.Textbox(value="generate a sexy red women dress",label="输入prompt")
                    i2i_gallery_1 = gr.Gallery(value=self.current_image_ls_1, label='图库展示1',height='auto',columns=3)
                    image_btn_1 =gr.UploadButton("上传至图库1", variant="primary", file_types=['image'])
                    i2i_gallery_2 = gr.Gallery(value=self.current_image_ls_1, label='图库展示2',height='auto',columns=3)
                    image_btn_2 =gr.UploadButton("上传至图库2", variant="primary", file_types=['image'])
                    process_btn = gr.Button("生成图片", variant='primary')
                with gr.Column():
                    i2i_gallery_output = gr.Gallery(label='结果图库',height='auto',columns=1)
                    status = gr.Markdown("等待选择...")
                    

            # 事件绑定
            add_btn.click(
                fn=self.gen_text_v2,
                inputs=[user_input, role_input, task_input, model_options, options_state],
                outputs=[options_state, dynamic_checkbox]  # 同时更新状态和显示组件
            )
            
            save_btn.click(
                fn=save_selected,
                inputs=[dynamic_checkbox, options_state],
                outputs=[output_msg, t2i_checkbox]
            )

            clear_btn.click(
                fn=delete_datafile,
                outputs=output_msg
            )

            image_btn.click(self.gen_text2Image,
                            inputs=[t2i_checkbox, t2i_prompt_input, mode_dropdown, t2i_model_dropdown, t2i_image_num_input, t2i_infer_steps, current_image_states],
                            outputs=[t2i_gallery,i2i_gallery_1, i2i_gallery_2,t2i_output_msg])
            


            def gallery_select_imageEditor(evt: gr.SelectData, selected_indices, current_image_states):
                """处理图片选择/取消选择"""
                
                print("selected_indices: ", selected_indices)
                index = evt.index
                print("evt: ", evt.index)
                image_path = current_image_states[index][0]
                # current = selected_indices.value.copy()
                res= { 'background': None, 'layers':[image_path],  'composite':None}
                return res
        
            def predict(im):
                return im["composite"]
            
            def cut_background(image):
                image = image["composite"]
                # img = Image.open(image)
                print("Cutting image...")
                img = self.agent.cut_image(image)
                print(" image cutted")
                return img
            
            def append_img_gallery(im_name, image, current_image_states):
                if not isinstance(image, str):
                    uint8_arr = image.astype(np.uint8)
                    # 生成 PIL 图像
                    image = Image.fromarray(uint8_arr)
                img_path = os.path.join(self.image_result, im_name) 
                image.save(img_path)
                current_image_states.append(img_path)
                print("append_img_gallery: ",current_image_states)
                return current_image_states

            
            t2i_gallery.select(
                fn=gallery_select_imageEditor,
                inputs=[selected_indices_0, t2i_gallery],
                outputs=im,
                show_progress=False
            )
            im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")

            cut_bg_btn.click(
                fn=cut_background,
                inputs=im,
                outputs=im_preview,
                show_progress=False
            )
            save_im_btn.click(
                fn=append_img_gallery,
                inputs=[im_name_text ,im_preview, current_image_states],
                outputs=i2i_gallery_1,
                show_progress=False
            )

            # step 3 logic
            i2i_gallery_1.select(
                fn=gallery_select,
                inputs=selected_indices_1,
                outputs=selected_indices_1,
                show_progress=False
            )

            i2i_gallery_2.select(
                fn=gallery_select,
                inputs=selected_indices_2,
                outputs=selected_indices_2,
                show_progress=False
            )

            # image_btn_1.click( )

            process_btn.click(
                fn=process_selected_images,
                inputs=[i2i_prompt_input, selected_indices_1, current_image_states_1,selected_indices_2, current_image_states_2 ],
                outputs=[i2i_gallery_output, status]
            )

        return demo


    def text2image_module(self, title='文生图', desc='文生图'):
        def load_data(existing_options):
            import pandas as pd
            data_path = os.path.join(self.text_result, "selected_sentences.csv") 
            df = pd.read_csv(data_path, header=0)
            print(df.head(10))
            word_ls = list(df['result'].values)
            return existing_options, gr.update(choices=word_ls + existing_options)

        options_state = gr.State(value=self.current_options)
        with gr.Blocks() as demo:
            # 文生图模块
            gr.Markdown(f"## {title}")
            with gr.Row():
                with gr.Column():
                    t2i_checkbox = gr.CheckboxGroup(
                        label="可选项目",
                        choices=self.current_options,
                        interactive=True,
                        elem_classes=["vertical-checkbox"]
                    )
                    t2i_prompt_input = gr.Textbox(label="输入prompt")
                    t2i_model_dropdown= gr.Dropdown(["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev", 'modelscope'], label='模型选择')
                    t2i_image_num_input = gr.Textbox(label="生成图片数量")
                    t2i_infer_steps = gr.Slider(0, 10, value=1, label="推理步数", info="similarity between 1 and 10")
                    t2i_update_btn = gr.Button("加载词", variant="primary")
                    image_btn = gr.Button("图片生成", variant="primary")
                with gr.Column():
                    t2i_gallery = gr.Gallery(label='图片展示',height='auto',columns=3)

            t2i_update_btn.click(load_data, inputs=[options_state], outputs=[options_state, t2i_checkbox])
            # image_btn.click(self.gen_text2Image, inputs=[t2i_prompt_input, t2i_model_dropdown, t2i_image_num_input, t2i_infer_steps],outputs=[t2i_gallery])
        return demo

    # def text_2_image_module_v2(self, title='文生图', desc="AI生成关键词"):
    #     with gr.Blocks() as demo:
    #         gr.Markdown(f"## {title}")
            
    #         # 状态存储
    #         options_state = gr.State(value=self.current_options)
    #         with gr.Row():
    #             # 添加新选项区域
    #             with gr.Column():
    #                  # role "你是一个时尚服装行业的专家"
    #     #task "请回答下面问题,只罗列答案不要返回多余的词,并以列表形式："
    #                 # new_option_input = gr.Textbox(label="输入Prompts", placeholder="输入要添加的内容...")
    #                 role_input = gr.Textbox(value="你是一个时尚服装行业的专家,请回答下面问题",label="输入角色", placeholder="输入要添加的内容...")
    #                 task_input = gr.Textbox(value= "答案要求:只罗列答案不要返回多余的词,并以列表形式返回, 每条列表格式如下:<序号>.<答案>", label="输入任务", placeholder="输入要添加的内容...")
    #                 user_input = gr.Textbox(value= "列出2024年最受欢迎的10个衣服品牌", label="输入任务", placeholder="输入要添加的内容...")
    #                 model_options = gr.Dropdown(['deepseek-r1:1.5b', 'llama3.2:latest','Qwen'], label='模型选择')
    #                 add_btn = gr.Button("➕ 生成文本", variant="primary")
    #                 clear_btn = gr.Button("重置数据文件", variant="primary")
                    

    
    

    def text_2_image_module(self, title='文生图', desc="AI生成关键词"):
        comp = gr.Interface(
                fn= self.gen_text2Image,
                inputs=[gr.Textbox(label="文本输入"),
                        gr.Dropdown(["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev", 'modelscope'], label='模型选择'),
                        gr.Textbox(label="生成图片数量"),
                        gr.Slider(0, 10, value=1, label="推理步数", info="similarity between 1 and 10")],
                outputs=[gr.Gallery(label='图片展示',height='auto',columns=3)],
                title=title,
                description=desc,
                theme="huggingface",
                examples=[
                    ["a girl in long beautiful white dress","FLUX.1-dev",1, 2],
                      ["生成冬装服装搭配","FLUX.1-dev",3,3],
                      ],
                cache_examples=False,
            )
        return comp

    def generate_interface(self,):
        tab_interface_ls = {}
        # module 1: 生词
        tab_interface_ls['AI生词'] = self.text_module_v2() #self.text_module()
        # module 2： 文生图
        tab_interface_ls['文生图'] = self.text2image_module() #self.text_2_image_module(title='文生图')

        # module 2: 服装上身
        tab_interface_ls['服装搭配'] = self.image_module('dress_up', title="服装搭配")
           
        # module 3: 换模特
        tab_interface_ls['更换模特'] = self.image_module('update_model', title="更换模特")

        comp = gr.TabbedInterface(
                list(tab_interface_ls.values()), list(tab_interface_ls.keys())
            )
        return comp

def main():
    print(f"Runing Gradio APP")
    component = GradioApp()
    component.generate_interface().launch(share=True)


if __name__ == "__main__":
    main()