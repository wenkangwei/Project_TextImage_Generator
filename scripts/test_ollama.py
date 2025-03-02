


######## 魔方ModelScope 线上模型部署请求
#参考： https://modelscope.cn/my/modelService/deploy?page=1&type=platform
#
####################
# 
# 
# from openai import OpenAI

# model_id = 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF'

# client = OpenAI(
#     base_url='https://ms-fc-900604be-9c62.api-inference.modelscope.cn/v1',
#     api_key='e37bfdad-0f6a-46c2-a7bf-f9dc365967e3'
# )

# response=client.chat.completions.create(
#     model=model_id,
#     messages=[{"role":"user", "content":"你好，能帮我介绍一下杭州吗？"}],
#     stream=True
# )

# for chunk in response:
#     print(chunk.choices[0].delta.content, end='', flush=True)


###############
# 备注： ollama 请求前要在 terminal里面用命令启动： ollama serve
#  之后在用API 请求
##############
# # Ollama 本地模型部署请求
# import requests
# import json
# url = "http://localhost:11434/api/generate"
# data= {"model":"deepseek-r1:1.5b","prompt":"Tell me a joke"}
# data_js = json.dumps(data)
# # print("data_js:", data_js)
# response = requests.post(url, json=data)
# if response.status_code == 200:
#         try:
#             data = response.json()
#             print("响应数据:", data)
#         except ValueError:
#             # 
#             # print("响应不是有效的 JSON:", response.text)
#             res= []
#             for t in response.text.split("\n"):
#                 print(t)
#                 try:
#                     res.append(json.loads(t)['response'])
                    
#                 except:
#                      pass
#             print("text data: ", ''.join(res))


#### Ollama 本地API请求

import ollama
from ollama import chat
from ollama import ChatResponse

# ollama.pull("deepseek-r1:1.5b")
# print( 'ollama result:',ollama.list())
response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)