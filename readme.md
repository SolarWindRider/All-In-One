# ALL-IN-ONE: A Reproduction of HuggingGPT

# 欧应万：复现HuggingGPT

由于chatGPT在国内无法合法访问，且访问huggingface的仓库容易掉线，而下载并运行这些模型会耗费大量资源。

本项目使用chatGLM替换chatGPT，使用百度Ai的api替换huggingface的各个模型，实现了对HuggingGPT的复现。

优点是占用资源小，响应速度快。

缺点是功能相对少，因为百度Ai的模型显然是比huggingface的少。

<div align="center">
<img src= ./assets/all-in-one.png width=10%>
</div>

## HuggingGPT Lite的复现

在对chatGLM-6b-int4-qe做了很长时间（2 days）的prompt测试后，发现它的效果很烂，完全不能实现任务拆分功能，人工智障一个。我的个人意见是，finetune肯定不能实现我要的效果，但是我目前能用的计算资源根本没法支持我对LLM做finetune，所以先用chatGPT复现HuggingGPT Lite。
