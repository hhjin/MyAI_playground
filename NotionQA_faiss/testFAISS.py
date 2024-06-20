from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
import pickle
import time
from openai import  InvalidRequestError 
from IPython.display import display, Markdown

import mylangchainutils

########## test 1 
pklName ="Indexs_backup/faiss_storeGPT4AGI.pkl"
indexFile="Indexs_backup/notionGPT4AGI.index"
store=mylangchainutils.loadFAISStore( indexFile,pklName)
index=store.index
 
mylangchainutils.similarity_search_with_score ( store ,"How to prompt productivity by AI tools?",2)
#mylangchainutils.similarity_search_with_score ( store ,"gpt-4-0125-preview limitation" )  # the default n=1 used while missing
 
# Load another test index of 5M
store5m = mylangchainutils.loadFAISStore( "Indexs_backup/notionDB_001.index", "faiss_storeMy.pkl")
index5m =store5m.index
 


########## test tweets notion db
storetweets = mylangchainutils.loadFAISStore( "Indexs_backup/notion_tweets.index", "Indexs_backup/faiss_store_tweets.pkl")
indextweets =storetweets.index
 
#matchedDocs=storetweets.similarity_search_with_score("How to prompt productivity by AI tools?",3)
#matchedDocs=storetweets.similarity_search_with_score("gpt-4-0125-preview limitation",2)
#printMatchedDocs (matchedDocs)


#mylangchainutils.similarity_search (storetweets ,"How to prompt productivity by AI tools?" )



llmAzure = AzureOpenAI(
    deployment_name="gpt35turbo",
    #model_name="gpt-3.5-turbo",
    temperature= 0.26
)






 

 

