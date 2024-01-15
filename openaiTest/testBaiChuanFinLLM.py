import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import PeftModel, PeftConfig

'''
# Baichuan2-13B-Chat-4bits
tokenizer = AutoTokenizer.from_pretrained("d:/models/Baichuan2-13B-Chat-4bits",
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True)

#在 textgenerationwebui 加载时，需要选择 transformers,   trust_remote_code , 12G GPU 用满

model = AutoModelForCausalLM.from_pretrained("d:/models/Baichuan2-13B-Chat-4bits",
    revision="v2.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("d:/models/Baichuan2-13B-Chat-4bits", revision="v2.0")
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)


'''

# Baichuan-13B-Chat   DISC-FinLLM finetue

tokenizer = AutoTokenizer.from_pretrained("D:/text-generation-webui003\models\DISC-FinLLM2", use_fast=False, trust_remote_code=True)

# 加载模型 原始精度模型 , OOM with 12G GPU
#model = AutoModelForCausalLM.from_pretrained("D:\models\Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

#加载quantize模型 开发者可以按照自己的需求修改模型的加载方式，但是请注意：如果是为了节省显存而进行量化，应加载原始精度模型到 CPU 后再开始量化；
#避免在from_pretrained时添加device_map='auto'或者其它会导致把原始精度模型直接加载到 GPU 的行为的参数
model = AutoModelForCausalLM.from_pretrained("D:/text-generation-webui003\models\DISC-FinLLM2",  torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(4).cuda()

# #在 textgenerationwebui 加载DISC-FinLLM2时，需要选择  load-in-4bit, double_quant, trust_remote_code

# load lora , 需要把基础模型的 generation_config.json copy to lora path
model.generation_config = GenerationConfig.from_pretrained("D:/text-generation-webui003\models\DISC-FinLLM2\Baichuan-13B-Chat-lora-Retrieval")
messages = []
messages.append({"role": "user", "content": "请解释一下什么是银行不良资产？"})
response = model.chat(tokenizer, messages)
print(response)

 
 

