
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain.embeddings import CohereEmbeddings
import sys
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
  
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader


# 在Cursor 中运行时， file_path 相对路径是 从当前workspace根目录(py)
loader = CSVLoader(file_path='pg_udtt300.csv' ,
    encoding="utf-8"
)
document_list = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2300, separator=". ")
                #RecursiveCharacterTextSplitter  try ??
 
'''````
docs = []
metadatas = []
for i, d in enumerate(document_list):
    splits = text_splitter.split_documents([d])
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))
''' 

# split it into chunks

docs = text_splitter.split_documents(document_list)

print(len(document_list))
size=len(docs)
print(size)

'''
#langchain_embeddings_Sentensetr = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#langchain_embeddings_Cohere  = CohereEmbeddings(cohere_api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS")

### For azure limited rate on embedding, swith to pure open ai before using OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings ,AzureOpenAIEmbeddings
langchain_embeddings_azureopenai = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")



# generate docs  embedding into Chroma
storevec = Chroma.from_documents(docs[:10], langchain_embeddings_azureopenai , persist_directory="./Chroma_DBtest")
'''


query = "What is new in each UDTT versions?"

# use my utils to query/print result with format 
#docsMatched =storevec.similarity_search(query)
 
myQAKit=QA_Toolkit("./Chroma_DBtest")
storeOpenAI_Chroma=myQAKit.get_dbstore_openai()
docs = myQAKit.similarity_search(storeOpenAI_Chroma, query, k=3) 

#storevec.persist()


