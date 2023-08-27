import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import sys
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
from langchain.embeddings.openai import OpenAIEmbeddings
from supabase.client import Client, create_client
from langchain.vectorstores import SupabaseVectorStore
supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
langchain_embeddings_azureopenai = OpenAIEmbeddings(deployment="text-embedding-ada-002")
#langchain_embeddings_azureopenai = OpenAIEmbeddings()

'''        ################# 术语说明 ###############

AIDoc           subapase project :  表名： documents            由gpt35pgvector在线Web创建的没有title, size的
===
BTT_DocumentAI  subapase project :  表名： pg, pg_1000, pg_430  由paul-graham-gpt 创建， 其中 pg 包含paul-graham article 和pg size 1600?
===
AI_Documents1.0GA  和 gpt35pgvector_raw 指的是 prompt 类型。   AI_Documents1.0GA  是GA 1.0定版prompt
===
'''


 
#从头生成 表数据 
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader

# 在Cursor 中运行时， file_path 相对路径是 从当前workspace根目录(py)
loader = CSVLoader(file_path='pg_udtt300.csv' ,
    encoding="utf-8"
)
document_list = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, separator=". ")
                #RecursiveCharacterTextSplitter  try ??
 
docs = []
metadatas = []
for i, d in enumerate(document_list):
    splits = text_splitter.split_documents([d])
    docs.extend(splits)
    metadatas.extend([{"source": "https://unicomglobal.com/btt/10.4/"}] * len(splits))

print(len(document_list))
print(len(metadatas))
size=len(docs)
print(size)       

 
vector_store = SupabaseVectorStore.from_documents([docs[0]],   embedding=langchain_embeddings_azureopenai, client=supabase, table_name="documents_langchain", query_name="match_documents_langchain")
matched_docs=vector_store.similarity_search("What is UDTT?")
print(matched_docs[0].page_content)
# end 从头生成 表数据 
 


# We're using the default `documents` table here.
#  You can modify this by passing in a `table_name` argument to the `from_documents` method.

query="What is new in  UDTT 10.4?"
vector_store = SupabaseVectorStore( embedding=langchain_embeddings_azureopenai, client=supabase, table_name="documents_langchain", query_name="match_documents_langchain")
matched_docs=vector_store.similarity_search(query)
print(matched_docs[0].page_content)
print(matched_docs[0].metadata)


retriever = vector_store.as_retriever(search_type="mmr")
matched_docs = retriever.get_relevant_documents(query)
for i, d in enumerate(matched_docs):
    print(f"\n\n\n## MMR searched Document {i}\n")
    print(d.page_content)
    print(matched_docs[0].metadata)
