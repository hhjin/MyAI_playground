
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter ,RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain.embeddings import CohereEmbeddings

import sys
sys.path.append('./utils')
from mylangchainutils import QA_Toolkit
 
# Load Notion page as a markdownfile file
from langchain.document_loaders import NotionDirectoryLoader
 
dbPath="/Users/henryking/Documents/MyObsidianNotes/Notion_DB/"
loader = NotionDirectoryLoader(dbPath)
documents = loader.load()
md_file=documents[0].page_content
print("#### raw documents counts:",len(documents))

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print("#### splitted docs counts:",len(docs))

# Test different LLM embedding functions
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

azure_openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="de1c603ee9a84d3aa0c0b82ccbdde577",
                api_base="https://bttaidoc.openai.azure.com/",
                api_type="azure",
                
                model_name="text-embedding-ada-002"
            )
cohere_ef  = embedding_functions.CohereEmbeddingFunction(api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS",  model_name="large")

cohere_efNLS  = embedding_functions.CohereEmbeddingFunction(
        api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS", 
        model_name="multilingual-22-12")

multilingual_texts  = [ 'Hello from Cohere!', 'مرحبًا من كوهير!', 
        'Hallo von Cohere!', 'Bonjour de Cohere!', 
        '¡Hola desde Cohere!', 'Olá do Cohere!', 
        'Ciao da Cohere!', '您好，来自 Cohere！',
        'कोहेरे से नमस्ते!'  ]


embeddingsBySenteTrsf            =sentence_transformer_ef(texts=["document1","document2"])
embeddingsByCohere               =cohere_ef(texts=["document1","document2"])
embeddingsByOpenAIazure          =azure_openai_ef(texts=["document1"])  # for azure rate limitation, NO concurrent embed_documents size>1
embeddingsBy_cohere_multilingual =cohere_efNLS(texts=multilingual_texts)

print(f"  sentence_transformer embedding size = {len(embeddingsBySenteTrsf[0])}")
print(f"  azure_openai embedding size = {len(embeddingsByOpenAIazure[0])}")
print(f"  Cohere embedding size = {len(embeddingsByCohere[0])}")
print(f"  Cohere NLS embedding size = {len(embeddingsBy_cohere_multilingual[0])}")
 
# langchain_embeddings object which is used as input param to generate Vector for Store

langchain_embeddings_Sentensetr = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
langchain_embeddings_Cohere  = CohereEmbeddings(cohere_api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS",
                                                model="multilingual-22-12" )

### For azure limited rate on embedding, swith to pure open ai before using OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
langchain_embeddings_azureopenai = OpenAIEmbeddings(deployment="text-embedding-ada-002")

''' 

###################  Only call once when creating DB !!!!!   ############################

print("\n\n\n\n$$$$$$$$$$$$$$$%%%$%$#!%@$#%@#%#@% Creating Chroma DB....")
storevec = Chroma.from_documents(docs, langchain_embeddings_Cohere , persist_directory="./Chorma_NotionDB/Cohere")
storevec.persist()
'''


# read Chroma DB as vector store

myQAKit=QA_Toolkit("./Chorma_NotionDB")

storevecRestored=myQAKit.get_dbstore_cohere( model="multilingual-22-12")
 
print("\n\n\n\n$$$$$$$$$$$$$$$%%%$%$#!%@$#%@#%#@% after read from persistence Chroma DB....")

query = "How to grow fast myself by learnning?"

while True:
        query =input("\n\n####### Query for notionDB: ")
        if query=="":
                break
        # use my utils to query/print result with format 
        #docsMatched =storevecRestored.similarity_search(query)
        docsMatched =myQAKit.similarity_search_with_score ( storevecRestored, query,2)

        print("\n\n\n\n$$$$$$$$$$$$$$$%%%$%$#!%@$#%@#%#@%  MMR get_relevant_documents ")

        retrieverMMR = storevecRestored.as_retriever(search_type="mmr" ,
                                                #search_kwargs={"k": 1} 
                                                )
        docs = retrieverMMR.get_relevant_documents(query)
        myQAKit.printMatchedDocs(docs)


