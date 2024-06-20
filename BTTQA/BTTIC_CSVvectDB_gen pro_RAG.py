import os

import sys
import json
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 


import re
from langchain.vectorstores import Chroma
import tiktoken

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

CHUNK_SIZE=  1666  #1666 #char   #334      #374   #430   #490
OVERLAP_SIZE= 166  #166 #char    #60       #123   #60    #90
 
tokenTextSplitter =  TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
charTextSplitter =  CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE ,separator=" ")

textSplitter=charTextSplitter  #charTextSplitter 用于1666c 字符切分
 
###### 如果 supabase 中断，可以从中断的indxe继续 ，##### Current block index  : 789
#contents = contents[1768:]
#titles= titles[1768:]
#urls= urls[1768:]
 

docs = []
metadatas = []
totalCount=0;
skipedCount=0
for i, d in enumerate(contents ):
    print("\n\n\n #### raw chunk " + str(i) + ":\n" +d)
 
    #取出 第一行的 context path
    lines = d.split('\n')
    first_line = lines[0].strip()
    context_path = first_line
    
    # 文本清洗  # pg_vector documents的清洗方式
    '''
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

    #### BLOCK FOR supabase  
    '''
    # 分小批量写入防止supabase报错，或者i是contents的最后一个索引，调用函数 SupabaseVectorStore.from_texts
    if (i + 1) % 2 == 0 or i == len(contents) - 1:
        print(f"\n\n\n ####### Total content size : {len(contents)}   ####### Current block index  : {i}")
        print(f"\n ####### Start to embedding block docs size  : {len(docs)}  - totalCount {totalCount}")

        ###################  Only call once when creating DB !!!!!   ############################    
        vector_store = SupabaseVectorStore.from_texts(docs,  metadatas=metadatas,  embedding=langchain_embeddings_azureopenai, client=supabase, table_name=table_name, query_name=query_name)
       
        # 清空docs和metadatas以准备下一批次
        totalCount=totalCount + len(docs)
        docs = []
        metadatas = []
    '''




print(f"\n\n ################################## total records : {totalCount}  {len(metadatas)} " )
print(f"\n\n ################################## skipedCount records : {skipedCount}  " )

 #### BLOCK FOR Chroma 
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

langchain_embeddings_Sentensetr = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5") # BAAI/bge-large-en-v1.5    #  maidalun1020/bce-embedding-base_v1
outputfile="udtt_ic1666c_bge-large-en-v1.5.csv"
with open(outputfile, 'a', encoding='utf-8') as f:
    f.write(f"id,content,metadata,embedding\n")

    for i, d in enumerate(docs):
        embed =langchain_embeddings_Sentensetr.embed_query(docs[i])
        txt=docs[i]
        txt=re.sub('"', '""', txt) 
       
        metadata=metadatas[i]
        metadata=json.dumps(metadata)
        metadata=re.sub('"', '""', metadata) 
        f.write(f'{i},"{txt}","{metadata}","{embed}"\n')

