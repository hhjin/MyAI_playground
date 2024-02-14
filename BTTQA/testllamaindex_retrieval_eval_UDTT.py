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
import sys
import time
import nest_asyncio
nest_asyncio.apply()
import asyncio

import json
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 

import re
from langchain.vectorstores import Chroma
import tiktoken

##############################################################################################
#       RAG 检索系统评估  hit-rate  MMR
#   UDTT 数据集， 数据装入处理清洗使用   langchain
#   VectorStore, ChromaDB 集成，评估使用 llama-index
#
#############################################################################################


#### !!!!! 重要  只运行一次 !!!!!!
create_new_ChromaDB    = False #Existing    # True  创建并储存 新的 ChromaDB   
create_new_qa_datasets = False #Existing    # True  创建并储存 新的 qa_datasets

####### 系统参数总控台 ： #######
splitBytoken=  True         # False by char  用于ic1666
CHUNK_SIZE=  334  #1666 #char   #334      #374   #430   #490  #2048
OVERLAP_SIZE= 60   #166 #char    #60       #123   #60    #90   #60

use_ChromaDB=  False   # False 使用 Memory store
model_Type="hf"     # hf azure  openai

hf_embeddingModel_Name="BAAI/bge-large-en"  # BAAI/bge-large-en  BAAI/bge-m3  BAAI/bge-large-en-v1.5  BAAI/bge-base-en-v1.5   maidalun1020/bce-embedding-base_v1

eval_chunks_num=366 # 在2千多个chunk中  评估多少个chunks
top_k=4

# when model_Type="openai" 
openai_chat_model_name="gpt-3.5-turbo"
openai_embedding_model_name="text-embedding-3-large"
# when model_Type="azure" or "hf" 
azure_chat_model_name="gpt35turbo-16k"
azure_embedding_model_name="embedding2"

model_name = hf_embeddingModel_Name.split('/')[-1]
chromadb_path="LocalData/chroma/UDTTIC"+str(CHUNK_SIZE)+"_llamaindex-eval-"+str(eval_chunks_num)+"/"+model_name
dataset_file="LocalData/datasets/UDTTIC"+str(CHUNK_SIZE)+"_eval_qa_"+str(eval_chunks_num)+".json"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


#从头生成 表数据 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter


#  file_path 相对路径是 从当前workspace根目录(py)
# Read the JSON file
with open('udtt_ic_by_urlist.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

contents = [item['content'] for item in json_data]
titles = [item['title'] for item in json_data]
urls = [item['url'] for item in json_data]
tokens= [item['tokens'] for item in json_data]
lengths= [item['length'] for item in json_data]

totalTokens=0
totalCharLength=0
for i, t in enumerate(tokens ):
    totalTokens+=t
    totalCharLength+=lengths[i]

print(f"###### totalTokens  : {totalTokens}")
print(f"###### totalCharLength  : {totalCharLength}")

tokenTextSplitter =  TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
charTextSplitter =  CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE ,separator=" ")

if splitBytoken:
    textSplitter=tokenTextSplitter   # charTextSplitter 用于ic1666
else:
    textSplitter=charTextSplitter 
 
docs = []
metadatas = []
skipedCount=0
for i, d in enumerate(contents ):
    print("\n\n\n #### raw chunk " + str(i) + ":\n" +d)
 
    #取出 第一行的 context path
    lines = d.split('\n')
    first_line = lines[0].strip()
    context_path = first_line
    
    ''' # 文本清洗  # pg_vector documents的清洗方式
    cleanedDoc = re.sub(r'\s+', ' ', d)   #把多个连续空格,换行制表符替换成单个空格 
    cleanedDoc=re.sub('\n', ' ', cleanedDoc)   #把所有换行替换成个空格 
    '''
    ## 保留换行的另一种清洗方式    
    cleanedDoc = re.sub(r' +', ' ', d)    # 把多个空格合并
    cleanedDoc=re.sub('\n{2,}', '\n', cleanedDoc)  # 把多个换行'\n\n'合并
    cleanedDoc=re.sub('\n +\n', '\n', cleanedDoc)  # 把多个换行空格换行合并
     
    #如果除了context path的内容很少, 跳过
    if   len(cleanedDoc) < len(context_path) + 160:
        print (lines[0]) 
        skipedCount+=1
        continue

    # 文本分割 
    splits  = textSplitter.split_text(cleanedDoc)

    # 检测 textSplitter 是否是TokenTextSplitter
    if isinstance(textSplitter, TokenTextSplitter):
        print("textSplitter is TokenTextSplitter")
        #检查splits 最后一个元素的token size ，倒数第二个元素与最后一个元素进行合并
        # only for token spliter
        token_count = num_tokens_from_string(splits[-1], "gpt-3.5-turbo")
        if token_count <OVERLAP_SIZE + 174  and len(splits) > 1:
            splits[-2] += splits[-1]
            splits.pop()
    else: 
        print("textSplitter is CharacterTextSplitter")
        char_count=len(splits[-1])
        if char_count <OVERLAP_SIZE + 666  and len(splits) > 1:
            splits[-2] += splits[-1]
            splits.pop()

    ## 拼接开头的title 和context path到 splits里
    for j in  range(len(splits)):
        if j==0 :
            #splits[j]=f"## Title: {titles[i]} - {j+1}\n## Context: {splits[j]}"
            splits[j]=f"-{j+1}  {splits[j]}"
        else   :
            #splits[j]=f"## Title: {titles[i]} - {j+1}\n## Context: {context_path}. {splits[j]}"
            splits[j]=f"-{j+1}  [{context_path}] {splits[j]}"
        
        print("\n\n #### handled chunk " + str(i) + " - " + str(j+1) + "  :\n" + splits[j])
    
    docs.extend(splits)
    metadatas.extend([{"source": urls[i] , "context_path": context_path}] * len(splits))

print(f"\n\n ################################## total records :    {len(metadatas)} " )
print(f" ################################## skipedCount records : {skipedCount}  " )

print ("\n########### dataset_file  :  " +dataset_file) 
if use_ChromaDB :
    print ("\n########### chromadb_path :  " +chromadb_path)      #"LocalData/chroma/UDTTIC490_llamaindex-AllChunks/openai-ada2-allrandomID"

from llama_index import Document
documents=[]
for i, doc in enumerate(docs ):
    document = Document(text=doc, metadata=metadatas[i])
    documents.append(document)

documents=documents[:eval_chunks_num]

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(documents):
    node.id_ = f"node_{idx}"

try:
    cache_folder= os.environ["TRANSFORMERS_CACHE"]
except KeyError as e:
	#raise Exception(f"Please set environment variables for {e.args[0]}")
    ...

from llama_index.llms import AzureOpenAI,OpenAI
from llama_index.embeddings import AzureOpenAIEmbedding ,OpenAIEmbedding

if model_Type=="azure" or model_Type=="hf": #### 缺省使用 Azure
    # 使用Azure 支持需要 .zprofile_azure , 由于已经把openai embedding 批量改成单步，所以修改embed_batch_size 以打印当前计数
    OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
    try:
        OPENAI_API_BASE= os.environ["OPENAI_API_BASE"]
    except KeyError as e:
        OPENAI_API_BASE=os.environ["AZURE_OPENAI_ENDPOINT"]
    OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]
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

start_time = time.time()
service_context = ServiceContext.from_defaults(llm=llm,  embed_model=embeddings_model )

if not use_ChromaDB :  # memory vector_index
    vector_index = VectorStoreIndex(documents, service_context=service_context )
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
        vector_index = VectorStoreIndex(documents, storage_context=storage_context, service_context=service_context )
    else :
        # 装入现存chromaDB 数据 , 注意 vector_index 构建方式的区别
        vector_index = VectorStoreIndex.from_vector_store( vector_store_chroma,  service_context=service_context)


retriever = vector_index.as_retriever(similarity_top_k=top_k)


if create_new_qa_datasets:
    # 初始化 qa_dataset， 只运行一次
    qa_dataset = generate_question_context_pairs(
        documents, llm=llm, num_questions_per_chunk=2
    )
    queries = qa_dataset.queries.values()
    #print(list(queries)[2])
    qa_dataset.save_json(dataset_file)
else :
    #  load qa_datase json
    qa_dataset = EmbeddingQAFinetuneDataset.from_json(dataset_file)

metrics = ["mrr", "hit_rate"]
include_cohere_rerank=False
if include_cohere_rerank:
    metrics.append(
        "cohere_rerank_relevancy"  # requires COHERE_API_KEY environment variable to be set
    )

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=retriever
)


# try it out on a sample query
sample_id, sample_query = list(qa_dataset.queries.items())[1]
sample_expected = qa_dataset.relevant_docs[sample_id]
print(" expected",sample_expected)
eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)
'''
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
print(f"\n#####   {modelName}   {dataset_file}   ###CHUNKSIZE : {CHUNK_SIZE}   ###eval_chunks_num: {eval_chunks_num}  ####topk :{top_k}  \n ")
if use_ChromaDB :
    if create_new_qa_datasets:
        print(f"\n##### Create new  chromadb_path: {chromadb_path}   \n ")
    else :
        print(f"\n##### Load existing chromadb_path: {chromadb_path}   \n ")

end_time = time.time()
run_time = end_time - start_time
print(f"\n###### cost time: %.2f 秒\n" % run_time)
print(display_results("top-2 eval result#", eval_results))