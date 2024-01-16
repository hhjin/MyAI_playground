from transformers import AutoTokenizer, AutoModel


# Only keep for trail to("mps"). Never got final output due to pending very very slow  with UserWarning: MPS: no support for int64 repeats mask, casting it to int32 
##tokenizer = AutoTokenizer.from_pretrained("/Users/henryking/models/chatglm2-6b", trust_remote_code=True)
##model = AutoModel.from_pretrained("/Users/henryking/models/chatglm2-6b",  trust_remote_code=True).to("mps")



## It is working by wait a minute. Ignore the error  :  'NoneType' object has no attribute 'cadam32bit_grad_fp32' \n Failed to load cpm_kernels:Unknown platform: darwin
tokenizer = AutoTokenizer.from_pretrained("/Users/henryking/models/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("/Users/henryking/models/chatglm2-6b-int4", trust_remote_code=True, device='cpu').float()

model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
