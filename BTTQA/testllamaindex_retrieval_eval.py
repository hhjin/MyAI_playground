from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SentenceSplitter

from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.evaluation import RetrieverEvaluator
import os
import nest_asyncio
nest_asyncio.apply()
import asyncio

#### !!!!! 重要  只运行一次 !!!!!!
create_new_ChromaDB    = False #Existing    # True  创建并储存 新的 ChromaDB   
create_new_qa_datasets = False #Existing    # True  创建并储存 新的 qa_datasets

model_Type="hf"     # hf azure  openai
hf_embeddingModel_Name="BAAI/bge-large-en-v1.5"  # BAAI/bge-m3  BAAI/bge-base-en-v1.5   maidalun1020/bce-embedding-base_v1   

use_ChromaDB=True      # False 使用 Memory store
# 注意 chromadb_path 要和 Model 匹配（ 当 use_ChromaDB== True）
chromadb_path="paul_graham_eval/Chroma_DB-bge-large"

qa_datasetsfile="paul_graham_eval/pg_eval_dataset3.json"

top_k=4
eval_chunks_num=0 # 评估多少个chunks

# when model_Type="openai" 
openai_chat_model_name="gpt-3.5-turbo"
openai_embedding_model_name="text-embedding-3-large"
# when model_Type="azure" or "hf" 
azure_chat_model_name="gpt35turbo-16k"
azure_embedding_model_name="embedding2"

documents = SimpleDirectoryReader("paul_graham_eval/raw_text").load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"
 # cut off the eval chunks size
if eval_chunks_num>0 :
    nodes=nodes[:eval_chunks_num]

from llama_index.llms import AzureOpenAI,OpenAI
from llama_index.embeddings import AzureOpenAIEmbedding ,OpenAIEmbedding
try:
    cache_folder= os.environ["TRANSFORMERS_CACHE"]
    OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
    try:
        OPENAI_API_BASE= os.environ["OPENAI_API_BASE"]
    except KeyError as e:
        OPENAI_API_BASE=os.environ["AZURE_OPENAI_ENDPOINT"]
    OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]
except KeyError as e:
	...
    
if model_Type=="azure" or model_Type=="hf": #### 缺省使用 Azure
    # 使用Azure 支持需要 .zprofile_azure , 由于已经把openai embedding 批量改成单步，所以修改embed_batch_size 以打印当前计数
    llm = AzureOpenAI(engine=azure_chat_model_name,  temperature=0.0, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
    embeddings_model=AzureOpenAIEmbedding(model=azure_embedding_model_name, embed_batch_size=100000, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
elif model_Type=="openai":
    # OpenAI 支持需要 .zprofile_openai
    llm=OpenAI(model=openai_chat_model_name)
    embeddings_model=OpenAIEmbedding(model=openai_embedding_model_name ,embed_batch_size=100000,)  #text-embedding-3-large  text-embedding-3-small

if model_Type=="hf":
    if cache_folder:
        embeddings_model = HuggingFaceEmbedding(model_name=hf_embeddingModel_Name ,cache_folder=cache_folder ) 
    else :
        embeddings_model = HuggingFaceEmbedding(model_name=hf_embeddingModel_Name )
        ...

service_context = ServiceContext.from_defaults(llm=llm,  embed_model=embeddings_model )

if not use_ChromaDB :  # memory vector_index
    vector_index = VectorStoreIndex(nodes, service_context=service_context )
else :
    # chromadb memory client
    #chroma_client = chromadb.EphemeralClient()
    #chroma_collection = chroma_client.create_collection("quickstart")
    # chromadb persistent client
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    vector_store_chroma = ChromaVectorStore(chroma_collection=chroma_collection)
    if create_new_ChromaDB :
        # 初始化DB，Save data,只运行一次！   ChromaVectorStore  index
        storage_context = StorageContext.from_defaults(vector_store=vector_store_chroma)
            #参考    #vector_index = VectorStoreIndex.from_documents( documents, storage_context=storage_context, service_context=service_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context )
    else :
        # 装入现存chromaDB 数据 , 注意 vector_index 构建方式的区别
        vector_index = VectorStoreIndex.from_vector_store( vector_store_chroma,  service_context=service_context)


retriever = vector_index.as_retriever(similarity_top_k=4)

if create_new_qa_datasets:
    # 初始化 qa_dataset， 只运行一次
    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=2
    )
    queries = qa_dataset.queries.values()
    #print(list(queries)[2])
    qa_dataset.save_json(qa_datasetsfile)
else :
    #  load qa_datase json
    qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_datasetsfile)

metrics = ["mrr", "hit_rate"]
include_cohere_rerank=False
if include_cohere_rerank:
    metrics.append(
        "cohere_rerank_relevancy"  # requires COHERE_API_KEY environment variable to be set
    )

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)

'''
# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[1]
sample_expected = qa_dataset.relevant_docs[sample_id]
print(" expected",sample_expected)
eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)

sample_id, sample_query = list(qa_dataset.queries.items())[2]
sample_expected = qa_dataset.relevant_docs[sample_id]
print(" expected",sample_expected)
eval_result = retriever_evaluator.evaluate(sample_query, ['node_42', 'node_57' ])
print(eval_result)
'''

async def evaluate_dataset(retriever, qa_dataset):
                                             
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    return eval_results

def run_evaluation(retriever, qa_dataset):
    loop = asyncio.new_event_loop()
    eval_results = loop.run_until_complete(evaluate_dataset(retriever, qa_dataset))
    return eval_results

import pandas as pd
def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    if include_cohere_rerank:
        crr_relevancy = full_df["cohere_rerank_relevancy"].mean()
        columns.update({"cohere_rerank_relevancy": [crr_relevancy]})

    metric_df = pd.DataFrame(columns)

    return metric_df

# try it out on an entire dataset
#eval_results =  await retriever_evaluator.aevaluate_dataset(qa_dataset)
eval_results = run_evaluation(retriever_evaluator, qa_dataset)

if  model_Type=="azure":
    modelName= azure_embedding_model_name
elif model_Type=="openai":
    modelName= openai_embedding_model_name
elif model_Type=="hf":
    modelName=hf_embeddingModel_Name
print(f"\n\n##### use_ChromaDB: {use_ChromaDB}  ###create_new_ChromaDB : {create_new_ChromaDB}   ###create_new_qa_datasets : {create_new_qa_datasets}  ")
print(f"\n#####   {modelName}   {qa_datasetsfile}  ####topk :{top_k}   ###eval_chunks_num: {eval_chunks_num}   \n ")

print(display_results("top-2 eval result#", eval_results))