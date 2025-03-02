# from transformers import AutoModelForCausalLM, AutoTokenizer

# from accelerate import Accelerator

# accelerator = Accelerator()


# # 加载模型和分词器
# model_name = "./deepseek-1.5b"  # 如果模型已下载到本地
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # 输入文本
# input_text = "你好，DeepSeek！"

# # 分词
# inputs = tokenizer(input_text, return_tensors="pt")

# # 推理
# outputs = model.generate(**inputs, max_length=50)

# # model, inputs = accelerator.prepare(model, inputs)

# # 解码输出
# output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("模型输出：", output_text)




#
# 参考： https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B?inference_provider=hf-inference&language=python&inference_api=true
#
#
#
#


from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="hf-inference",
	api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
	messages=messages, 
	max_tokens=500,
)

print(completion.choices[0].message)