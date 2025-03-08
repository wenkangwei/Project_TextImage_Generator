# 功能需求
前端
- [x] 生词模型界面
- [ ] 图片上传框
- [ ] 参考图选项
- [ ] 批量展示生成图
- [ ] 功能选项
- [ ] 多结果图展示
- [ ] 用户打标功能


后端
- [x] 接入 本地(ollama)/线上(ModelScope部署) 文本大模型
- [ ] 接入图片生成大模型（stable diffusion）
- [ ] 接入数据库存储 商品图片/打标信息
- [ ] 文本生成
- [ ] 图片生成
- [ ] 用户登录

# 问题
1. 网上的FLUX 对中文理解效果不好, 而且生成图片速度慢
2. 使用本地服务器+ 线上url请求远程模型形式 速度会比较慢
3. huggingface上面的部署的开源模型请求 在国内会被禁, 请求速度会很慢,需要梯子
4. 国内云服务收费比较贵, 现在用modelscope的开发环境使用, 但是对拉取huggingface模型不太友好,连不上海外网络


# 参考链接
- gradio official doc: https://www.gradio.app/guides/connecting-to-a-database
- gradio 构建用户登录界面： https://blog.csdn.net/lsqingfeng/article/details/132599955
- ollama 大模型部署: https://github.com/ollama/ollama-python
- gradio gallery参考 https://www.gradio.app/docs/gradio/gallery
- 绘蛙 https://www.ihuiwa.com/workspace/ai-image/one-shot
- Stable-diffusion-webui https://github.com/AlUlkesh/stable-diffusion-webui-images-browser/blob/main/scripts/image_browser.py
- kaggle stable diffusion: https://www.kaggle.com/code/wenkangw/stable-diffusion-webui-kaggle/edit
- stable-diffusion
    - 图生图参考: https://colab.research.google.com/github/stability-ai/stability-sdk/blob/main/nbs/SD3_API.ipynb#scrollTo=gj4YsYB8J4a7
    - stable diffusion 官网: https://platform.stability.ai/account/keys 