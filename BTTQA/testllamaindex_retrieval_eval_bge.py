from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SentenceSplitter

from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb

import os
import nest_asyncio
nest_asyncio.apply()
import asyncio

OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE=os.environ["OPENAI_API_BASE"]
OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]

documents = SimpleDirectoryReader("paul_graham_eval/").load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
# 使用Azure OpenAI Service
azure_lm = AzureOpenAI(engine="gpt35turbo-16k",  temperature=0.0, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
azure_embeddings=AzureOpenAIEmbedding(model="embedding2", azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5" )

service_context = ServiceContext.from_defaults(llm=azure_lm, embed_model=azure_embeddings )

 
#default memory vector_index
vector_index = VectorStoreIndex(nodes, service_context=service_context )
'''

# chromadb vector_index
# chromadb memory client
#chroma_client = chromadb.EphemeralClient()
#chroma_collection = chroma_client.create_collection("quickstart")

# chromadb persistent client
chroma_client = chromadb.PersistentClient(path="paul_graham_eval/Chroma_DB_test2")
chroma_collection = chroma_client.get_or_create_collection("quickstart")
vector_store_chroma = ChromaVectorStore(chroma_collection=chroma_collection)

# 装入现存chromaDB 数据 , 注意 vector_index 构建方式的区别
#vector_index = VectorStoreIndex.from_vector_store( vector_store_chroma,  service_context=service_context)

# 初始化DB，Save data,只运行一次！   ChromaVectorStore  index
storage_context = StorageContext.from_defaults(vector_store=vector_store_chroma)
  #参考    #vector_index = VectorStoreIndex.from_documents( documents, storage_context=storage_context, service_context=service_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context )
# end 初始化DB，只运行一次！

# end chromadb vector_index
'''

retriever = vector_index.as_retriever(similarity_top_k=2)

from llama_index.response.notebook_utils import display_source_node
retrieved_nodes = retriever.retrieve("What did the author do growing up?")
#retrieved_nodes = retriever.retrieve("How did Y Combinator challenge the traditional notion of \"deal flow\" and contribute to the creation of new startups that would not have existed otherwise?")
for node in retrieved_nodes:
    display_source_node(node, source_length=1000)
    print(node)

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

''' 初始化 qa_dataset， 只运行一次
qa_dataset = generate_question_context_pairs(
    nodes, llm=azure_lm, num_questions_per_chunk=2
)
queries = qa_dataset.queries.values()
print(list(queries)[2])
qa_dataset.save_json("paul_graham_eval/pg_eval_dataset.json")
'''
#  load qa_datase json
qa_dataset = EmbeddingQAFinetuneDataset.from_json("paul_graham_eval/pg_eval_dataset.json")

from llama_index.evaluation import RetrieverEvaluator

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

print(display_results("top-2 eval result#", eval_results))