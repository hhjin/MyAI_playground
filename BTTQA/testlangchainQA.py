import os
import openai
from langchain.llms import AzureOpenAI

openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = 'de1c603ee9a84d3aa0c0b82ccbdde577'

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

import faiss
from langchain.vectorstores import FAISS
from IPython.display import display, Markdown

from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import mylangchainutils
 

##################  Completion API of  Azure OpenAI  

llm = AzureOpenAI(
    deployment_name="gpt35turbo",
    model_name="text-davinci-002",
    temperature= 0,
    max_tokens=100,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
resp = llm("I am a robot")
 

storetweets = mylangchainutils.loadFAISStore( "Indexs_backup/notion_tweets.index", "Indexs_backup/faiss_store_tweets.pkl")
indextweets =storetweets.index

query="I am a robot"
#response = storetweets.query(query)
response = storetweets.query(query, llm=llm)
display(Markdown(response))

'''
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

response = qa_stuff.run(query)
'''


examples = [
    {
        "query": "Does Apple published new \
            AR VR devices?",
        "answer": "Yes"
    },
    {
        "query": "How to make use of  \
        AI for personal productivity?",
        "answer": "Make use of OpenAI for work flow and coding"
    }
]

