from llama_index.embeddings import AzureOpenAIEmbedding ,OpenAIEmbedding 
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import AzureOpenAI
import chromadb
import os
from llama_index.embeddings import HuggingFaceEmbedding

# 使用Azure OpenAI Service , 需要 llamaindex_src_change/embeddings/openai.py 覆盖   
try:
    cache_folder= os.environ["TRANSFORMERS_CACHE"]
    OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
    OPENAI_API_BASE=os.environ["OPENAI_API_BASE"]
    OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]
except KeyError as e:
	...

azure_lm = AzureOpenAI(engine="gpt35turbo-16k",  temperature=0.0, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
azure_embeddings=AzureOpenAIEmbedding(model="embedding2", azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)

if cache_folder:
    bge_embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v2-base-en" ,cache_folder=cache_folder )
else :
     #bge_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5" )
     ...

service_context = ServiceContext.from_defaults(llm=azure_lm , embed_model=azure_embeddings )

'''
from llama_index.llms import OpenAI
# .zporifile_openai needed
llm = OpenAI(model="gpt-3.5-turbo")
# not work for azure _get_query_embedding input engine : OpenAIEmbeddingModeModel.TEXT_EMBED_ADA_002
# service_context = ServiceContext.from_defaults(llm=azure_lm  ) 
service_context = ServiceContext.from_defaults(llm=llm) 
'''


# load from disk
#db2 = chromadb.PersistentClient(path="LocalData/chroma/Chroma_DB_UDTT_IC490_migrated/OpenAI")
#chroma_collection = db2.get_or_create_collection("langchain")

db2 = chromadb.PersistentClient(path="LocalData/chroma/UDTTIC490_llamaindex-AllChunks/openai-ada2-allrandomID")
chroma_collection = db2.get_or_create_collection("quickstart")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
vector_index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context
)
retriever = vector_index.as_retriever(similarity_top_k=5)
 
# Query Data from the persisted index
query_engine = vector_index.as_query_engine()
query="What is new in each UDTT versions?"
response = query_engine.query(query)
print(f"\n######## LLM answer for {query} \n\n",response)
 
retrieved_nodes = retriever.retrieve("What is new in each UDTT versions?")
for node in retrieved_nodes:
    print("\n\n ###### retrieved_nodes" , node)
