"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import time
from openai import  InvalidRequestError 


dbPath="Notion_DB/"
pklName ="faiss_storeMy.pkl"
indexFile="notionMy.index"
print (" source path: "+dbPath)
print (" indexFile: "+indexFile)
print (" pklName: "+pklName)
 

# Here we load in the data in the format that Notion exports it in.
ps = list(Path(dbPath).glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p ,encoding='utf8') as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1600, separator="\n")
docs = []
metadatas = []
 
indexList=None

for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


for i, d in enumerate(docs):
    if len(docs[i])>3000:
      print(docs[i])
      print("\n\n##################### Warnning !!!!!! , doc length exceeded !!!!!!, remove from docs list")
      docs.pop(i)
      metadatas.pop(i)


     

size=len(docs)

print(len(data))
print(len(docs))
print(len(metadatas))

# ##############   verify the azure api, ######### commended out in normal run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   !!!!!!   !!
#
'''
try:
    store0 = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
except  InvalidRequestError as e:
        print("\n\n############# OK to verify Azure  : ", e)
        time.sleep(10)
 

store0 = FAISS.from_texts(  [docs[0]] , OpenAIEmbeddings(), metadatas= [metadatas[0]])
indexList=store0.index

docs.pop(0) 
print(len(docs))
count=1
 
for i, d in enumerate(docs):
    try:
        delay=16
        count=count+1
        print("\n\n\n§∞§∞§¶∞¶¶§§∞∞§§¶∞¢∞¢§¶¶∞¢£¡™£¢∞¢¢££££££∞∞¢∞∞∞∞§§§ : ",count, "/",size,"  ###  sleep seconds :", delay)
        time.sleep(delay)
        print("\n\n")

        store = FAISS.from_texts(  [docs[i]] , OpenAIEmbeddings(), metadatas= [metadatas[i]])
        indexList.merge_from(store.index)
        #indexList.add((store.index))
    except InvalidRequestError as e:
        print("\n\n\n############# Error : ", e)
        time.sleep(delay)
        print("\n\n\n")
        continue
'''

store0 = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
indexList=store0.index
print("\n\n############### write index file "+indexFile)
faiss.write_index(indexList, indexFile)

print("\n\n############### write store file "+pklName)
store0.index = None
with open(pklName, "wb") as f:
    pickle.dump(store0, f)

