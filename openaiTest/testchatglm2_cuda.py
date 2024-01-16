from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True, device='cpu').float()

#tokenizer = AutoTokenizer.from_pretrained("g:/models/chatglm/chatglm2-6b", trust_remote_code=True)
#model = AutoModel.from_pretrained("g:/models/chatglm/chatglm2-6b", trust_remote_code=True).quantize(8).half().cuda()

#tokenizer = AutoTokenizer.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True)
#model = AutoModel.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True).half().cuda()

#tokenizer = AutoTokenizer.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True)
#model = AutoModel.from_pretrained("g:/models/chatglm/chatglm2-6b-int4", trust_remote_code=True).quantize(8).half().cuda() 

# It is OK for all aboves,  but OOM for chatglm2-6b without .quantize(8)

model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
