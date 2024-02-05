from sentence_transformers import SentenceTransformer

queries = ['query_1', 'query_2']
passages = ["sample document 1", "sample document 2"]
instruction = "为这个句子生成表示以用于检索相关文章："  # for both en an zh , if empty , it is more distinct

'''
model = SentenceTransformer('BAAI/bge-large-en')
#model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
print(scores)
'''

import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
cache_dir= '/Users/henryking/.cache/huggingface/hub'
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large' ,local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large',local_files_only=True)

#tokenizer = AutoTokenizer.from_pretrained('/Users/henryking/Desktop/BGE_reranker_large' )
#model = AutoModelForSequenceClassification.from_pretrained('/Users/henryking/Desktop/BGE_reranker_large')
#model.save_pretrained("/Users/henryking/Desktop/BGE_reranker_large")
end_time = time.time()
run_time = end_time - start_time
print(f"\n....load  model cost time: %.2f 秒\n" % run_time)

model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
