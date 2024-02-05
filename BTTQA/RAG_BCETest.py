from sentence_transformers import SentenceTransformer

'''
# list of sentences
sentences = ['sentence_0', 'sentence_1', ...]

# init embedding model
## New update for sentence-trnasformers. So clean up your "`SENTENCE_TRANSFORMERS_HOME`/maidalun1020_bce-embedding-base_v1" or "～/.cache/torch/sentence_transformers/maidalun1020_bce-embedding-base_v1" first for downloading new version.
model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")

# extract embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings)
'''



from BCEmbedding import RerankerModel
import time
start_time = time.time()
# your query and corresponding passages
query = 'input_query'
passages = ['passage_0', 'passage_1']

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# init reranker model , # 为了快速本地加载，需要修改RerankerModel 源码，AutoTokenizer 添加 local_files_only=True
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1",local_files_only=True)

end_time = time.time()
run_time = end_time - start_time
print(f"\n....load  model cost time: %.2f 秒\n" % run_time)

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)
print(scores)
# method 1: rerank passages 在 RerankerModel.rerank 方法中，当“通道”很长时，我们提供了一种先进的预处理程序，用于生产 sentence_pairs 。
rerank_results = model.rerank(query, passages)
print(rerank_results)
