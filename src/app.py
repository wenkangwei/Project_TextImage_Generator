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
import moviepy.editor as mppyth
from moviepy.editor import *
# from movie_generator.agi.suno.suno import Suno
import requests


import ollama
from ollama import chat
from ollama import ChatResponse

# ollama.pull("deepseek-r1:1.5b")
# print( 'ollama result:',ollama.list())
# response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

def call_LLM(inputs, prompts= '你是一个时尚服装行业的专家， 请回答下面问题：', model_version = 'Qwen'):
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
            'content': prompts + " " + inputs,
        },
        ])
        return response['message']['content']
    else:
        return "LLM version is not supported yet."
import os
class GradioApp:
    def __init__(self,config=None):
        #config with info of 
        # model version
        # prompts
        #others
        self.config=config
        # self.image_dir = "/mnt/d/workspace/projects/Project_TextImage_Generator/examples"
        self.image_dir = "../examples"
        self.model_dir = os.path.join(self.image_dir, "models")
        self.clothes_dir = os.path.join(self.image_dir, "clothes")
        self.reference_dir = os.path.join(self.image_dir, "references")
        self.model_files = [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir)]
        self.clothes_files = [os.path.join(self.clothes_dir, f) for f in os.listdir(self.clothes_dir)]
        self.reference_files = [os.path.join(self.reference_dir, f) for f in os.listdir(self.reference_dir)]
        pass
    
    
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
                theme="huggingface",
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
        return call_LLM(inputs,prompts, model_version=LLM_version)
    
    def text_module(self, title='文本生成', desc="AI生成关键词"):
        comp = gr.Interface(
                fn= self.gen_text,
                inputs=[gr.Textbox(label="文本输入"), gr.Dropdown(['deepseek-r1:1.5b', 'llama3.2:latest','Qwen'], label='模型选择')],
                outputs=[gr.Textbox(label="结果输出")],
                title=title,
                description=desc,
                theme="huggingface",
                examples=[
                    ["列出2024年最受欢迎的10个衣服品牌","llama3.2:latest"],
                      ["哪些款式的女装比较潮流， 请列出10个女装品类","Qwen"],
                      ["随机生成10个衣服类目并列出来","Qwen"]],
                cache_examples=True,
            )
        return comp
    
    def generate_interface(self,):
        tab_interface_ls = {}
        # module 1: 生词
        tab_interface_ls['AI生词'] = self.text_module()

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
    fire.Fire()




# class GradioUnitTest():
#     def __init__(self):
#         api_key =  "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"
#         # "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"
#         os.environ["OPENAI_API_KEY"] = api_key
#         self.llm_model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
#         # self.llm_model= None
#         self.client = OpenAI(api_key=api_key)
#         cur_path =os.getcwd()
#         root_path = '/'.join(cur_path.split("/")[:-2])

#         suno_result_path = os.path.join(root_path,'examples','suno_musics')
#         # self.suno = Suno(result_path=suno_result_path)
#         self.suno= None
#         pass
    
#     def test_text(self, input_text, mode = 'count'):
#         def process_test( _text, mode = 'count'):
#             def count_words(text):
#                 words = text.split(" ")
#                 res_dict = {}
#                 for word in words:
#                     if word in res_dict:
#                         res_dict[word] += 1
#                     else:
#                         res_dict[word] = 1
#                 res = "\n".join([f"word: {key}, count: {value}" for key, value in res_dict.items()])
#                 return res
            
#             def reverse_text(text):
#                 return text[::-1]
            
#             if mode == 'count':
#                 return count_words(_text)
#             return reverse_text(_text)

#         res = f"Got text from textbox: {input_text}"
#         return res, process_test(input_text, mode)
#         # return res, count_words(input_text)
    
#     def test_image(self, input_image, filter_mode='sepia'):
#         def filter_image(input_image, filter_mode='sepia'):
#             def sepia(input_img):
#                 sepia_filter = np.array([
#                     [0.393, 0.769, 0.189], 
#                     [0.349, 0.686, 0.168], 
#                     [0.272, 0.534, 0.131]
#                 ])
#                 sepia_img = input_img.dot(sepia_filter.T)
#                 sepia_img /= sepia_img.max()
#                 return sepia_img
#             def grayscale(input_img):
#                 input_img = np.mean(input_img, axis=2) / np.max(input_img)
#                 return input_img
            
#             if filter_mode == 'sepia':
#                 return sepia(input_image)
#             elif filter_mode == 'grayscale':
#                 return grayscale(input_image)
#             else:
#                 return input_image
#         res = f"Got image from image input: {input_image}"
#         filtered_image = filter_image(input_image, filter_mode)
#         return res, filtered_image
    
#     def test_audio(self, input_audio, filter_mode='echo', prompt='', checkbox_ls=[]):
#         def process_audio(input_audio, filter_mode='echo'):
#             print("input_audio shape: ", input_audio[1].shape, input_audio)
#             def echo(input_audio):
#                 aud = np.concatenate([input_audio[1], input_audio[1]], axis=0)
#                 return (input_audio[0], aud)
#             def reverse(input_audio):
#                 return (input_audio[0], input_audio[1][::-1]) 
            
#             if filter_mode == 'echo':
#                 res_audio = echo(input_audio)
#             elif filter_mode == 'reverse':
#                 res_audio = reverse(input_audio)
#             else:
#                 res_audio = input_audio
#             return res_audio
#         print("checkbox_ls: ", checkbox_ls)
#         res = f"Got audio from audio input: {input_audio}"
#         wait_audio = 'wait_audio' in checkbox_ls
#         make_instrumental = 'make_instrumental' in checkbox_ls
#         if checkbox_ls != []:
#             print('checlbox_ls: ', checkbox_ls)
#         generated_audio_path=''
#         if prompt != '':
#             music_paths = self.test_music_generation(prompt, make_instrumental, wait_audio)
#             generated_audio_path = '\n'.join(music_paths)
#             res = f"Got audio from suno: {generated_audio_path}"
#         processed_audio = process_audio(input_audio, filter_mode)
#         return res, processed_audio, generated_audio_path

#     def test_video(self, input_video, filter_mode='flip'):
#         def process_video(input_video, filter_mode='flip'):
#             print("input_video data: ", input_video)

#             def clip(input_video):
#                 clip1 = VideoFileClip(input_video)
#                 clip2 = VideoFileClip(input_video).subclip(2,3)
#                 clip3 = VideoFileClip(input_video)
#                 final_clip = concatenate_videoclips([clip1,clip2,clip3])
#                 output_video = "final_clip.mp4"
#                 final_clip.write_videofile(output_video)
#                 return output_video
#             def flip(input_video):
#                 return np.flip(input_video, axis=1)
#             def rotate(input_video):
#                 return np.rot90(input_video)
#             if filter_mode == 'clip':
#                 return clip(input_video)
#             elif filter_mode == 'flip':
#                 return flip(input_video)
#             elif filter_mode == 'rotate':
#                 return rotate(input_video)
#             else:
#                 return input_video
#         res = f"Got video from video input: {input_video}"
#         processed_video = process_video(input_video, filter_mode)
#         return res, processed_video

#     def test_chatbot(self, input_text, history):
#         history_langchain_format =[]
#         for human, ai in history:
#             history_langchain_format.append(HumanMessage(human))
#             history_langchain_format.append(AIMessage(ai))
#         history_langchain_format.append(content=input_text)
#         llm_response = self.llm_model(history_langchain_format)
#         return llm_response.content

#     def predict(self, message, history):
#         history_openai_format = []
#         for human, assistant in history:
#             history_openai_format.append({"role": "user", "content": human })
#             history_openai_format.append({"role": "assistant", "content":assistant})
#         history_openai_format.append({"role": "user", "content": message})
    
#         response = self.client.chat.completions.create(model='gpt-3.5-turbo',
#         messages= history_openai_format,
#         temperature=1.0,
#         stream=True)

#         partial_message = ""
#         for chunk in response:
#             if chunk.choices[0].delta.content is not None:
#                 partial_message = partial_message + chunk.choices[0].delta.content
#                 yield partial_message
    
#     def predict_v2(self, message, history):
        
#         url = "https://api.link-ai.chat/v1/chat/completions"
#         headers = {
#             'Authorization': 'Bearer Link_USN4Vru40ciqYkdpeWywmOOIOPHGLYm8EuAGm0xE0b',
#             'Content-Type': 'application/json'
#         }
#         history_openai_format = []
#         for human, assistant in history:
#             history_openai_format.append({"role": "user", "content": human })
#             history_openai_format.append({"role": "assistant", "content":assistant})
#         history_openai_format.append({"role": "user", "content": message})
    

#         data = {
#             "app_code": "default",
#             "messages": history_openai_format,
#         }

#         response = requests.post(url, headers=headers, json=data).json()
#         partial_message = ""
#         for chunk in response['choices']:
#             if chunk['message']["content"] is not None:
#                 partial_message = partial_message + chunk['message']["content"]
#                 yield partial_message
    

#     def predict_v3(self, message, history):
        
#         url = "https://api.link-ai.chat/v1/chat/completions"
#         headers = {
#             'Authorization': 'Bearer Link_USN4Vru40ciqYkdpeWywmOOIOPHGLYm8EuAGm0xE0b',
#             'Content-Type': 'application/json'
#         }
#         history_openai_format = []
#         for human, assistant in history:
#             history_openai_format.append({"role": "user", "content": human })
#             history_openai_format.append({"role": "assistant", "content":assistant})
#         history_openai_format.append({"role": "user", "content": message})
    

#         data = {
#             "app_code": "default",
#             "messages": history_openai_format,
#         }

#         response = requests.post(url, headers=headers, json=data).json()
#         partial_message = ""
#         for chunk in response['choices']:
#             if chunk['message']["content"] is not None:
#                 partial_message = partial_message + chunk['message']["content"]
#                 yield partial_message

#     def test_music_generation(self, prompt, make_instrumental=False, wait_audio=False):
#         request = {
#             "prompt": prompt,
#             "make_instrumental": make_instrumental,
#             "wait_audio": wait_audio
#             }
#         # music_ls = self.suno.generate_music(request)
#         music_ls = []
#         return music_ls

#     def run_test(self, mode='text'):
#         tab_interface_ls = {}
#         if mode == 'text' or mode == 'mix':
#             comp = gr.Interface(
#                 fn= self.test_text,
#                 inputs=['textbox', gr.Dropdown(['count', 'reverse'])],
#                 outputs=["textbox", "textbox"],
#                 title="test text module",
#                 description="test text.",
#                 theme="huggingface",
#                 examples=[
#                     ["A group of friends go on a road trip to find a hidden treasure."],
#                     ["A scientist discovers a way to travel through time."],
#                     ["A group of survivors try to escape a zombie apocalypse."],
#                 ],
#             )
#             tab_interface_ls['Text Ops'] = comp
#             if mode == 'text':
#                 return comp
#         if mode == 'image' or mode == 'mix':
#             # https://www.gradio.app/guides/the-interface-class
#             comp = gr.Interface(
#                 fn= self.test_image,
#                 inputs=['image', gr.Dropdown(['sepia', 'grayscale'])],
#                 outputs=["textbox",'image'],
#                 title="test image preprocess Module",
#                 description="test text.",
#                 theme="huggingface",
#                 examples=[
#                     ["/mnt/c/Users/wwk/Pictures/OIP.jpg", "sepia"],
#                 ],
#             )
#             tab_interface_ls['Image Ops'] = comp
#             if mode == 'image':
#                 return comp

#         if mode == 'audio' or mode == 'mix':
#             comp = gr.Interface(
#                 fn= self.test_audio,
#                 inputs=['audio', gr.Dropdown(['echo', 'reverse']), 'textbox', gr.CheckboxGroup([ 'make_instrumental' ,'wait_audio'],  label="Suno options", info="make_instrumental<bool>, wait_audio:<bool>") ],
#                 outputs=["textbox", 'audio'],
#                 title="test audio preprocess Module",
#                 description="test audio.",
#                 theme="huggingface",
#                 examples=[
#                     ["/mnt/d/workspace/projects/movie_generator/examples/audio/两只老虎，两只老虎-神秘-欢快-v2.mp3", "echo"],
#                     ["/mnt/d/workspace/projects/movie_generator/examples/audio/两只老虎，两只老虎-神秘-欢快-v2.mp3", "reverse"],
#                 ],
#             )
#             tab_interface_ls['Audio Ops'] = comp
#             if mode == 'audio':
#                 return comp
            
#         if mode == 'video' or mode == 'mix':
#             comp = gr.Interface(
#                 fn= self.test_video,
#                 inputs= [ 'video', gr.Dropdown(['clip', 'rotate'])],
#                 outputs=["textbox", 'video'],
#                 title="test video preprocess Module",
#                 description="test video.",
#                 theme="huggingface",
#                 examples=[
#                     ["/mnt/d/workspace/projects/movie_generator/examples/video/2月12日.mp4", "clip"],
#                 ],
#                 )
#             tab_interface_ls['Video Ops'] = comp
#             if mode == 'video':
#                 return comp
            
#         if mode == 'chat' or mode == 'mix':
#             # https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
#             # comp = gr.ChatInterface(self.test_chatbot)
#             comp = gr.ChatInterface(self.predict_v2)
#             tab_interface_ls['ChatBot'] = comp
#             if mode == 'chat':
#                 return comp    
#         if mode == 'mix':
#             # mix mode, use radio button to select the mode
#             comp = gr.TabbedInterface(
#                 list(tab_interface_ls.values()), list(tab_interface_ls.keys())
#             )
#             return comp
#         else:
#             def flip_text(x):
#                 return x[::-1]
#             def flip_image(x):
#                 return np.fliplr(x)
#             with gr.Blocks() as comp:
#                 gr.Markdown("Flip text or image files using this demo.")
#                 with gr.Tab("Flip Text"):
#                     text_input = gr.Textbox()
#                     text_output = gr.Textbox()
#                     text_button = gr.Button("Flip")
#                 with gr.Tab("Flip Image"):
#                     with gr.Row():
#                         image_input = gr.Image()
#                         image_output = gr.Image()
#                     image_button = gr.Button("Flip")

#                 with gr.Accordion("Open for More!", open=False):
#                     gr.Markdown("Look at me...")
#                     temp_slider = gr.Slider(
#                         minimum=0.0,
#                         maximum=1.0,
#                         value=0.1,
#                         step=0.1,
#                         interactive=True,
#                         label="Slide me",
#                     )
#                     temp_slider.change(lambda x: x, [temp_slider])

#                 text_button.click(flip_text, inputs=text_input, outputs=text_output)
#                 image_button.click(flip_image, inputs=image_input, outputs=image_output)
#         return comp