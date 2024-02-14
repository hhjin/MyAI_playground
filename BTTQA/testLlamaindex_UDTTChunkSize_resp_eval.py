import nest_asyncio
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
nest_asyncio.apply()

from llama_index import (  
    SimpleDirectoryReader,  
    VectorStoreIndex,  
    ServiceContext,  
)  
from llama_index.evaluation import (  
    DatasetGenerator,  
    FaithfulnessEvaluator,  
    RelevancyEvaluator ,
    ContextRelevancyEvaluator,
    AnswerRelevancyEvaluator
)  
from llama_index.llms import OpenAI
import os 
import time
from llama_index.llms import AzureOpenAI,OpenAI
from llama_index.embeddings import AzureOpenAIEmbedding ,OpenAIEmbedding


import json
import sys
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 

import re
from langchain.vectorstores import Chroma
import tiktoken

##############################################################################################
#       [[使用LlamaIndex评估RAG系统的理想块大小]] 用相关性和忠实性指标来评估 
#
# 统计结果，chunk-size 越大，检索hit-rate  MMR指标越好。 这种不收敛的结果，似乎表明了某种归纳偏见。这也是那篇文章用 用相关性和忠实性指标来评估的原因吧。
#   UDTT 数据集， 数据装入处理清洗使用   langchain
#   VectorStore, ChromaDB 集成，评估使用 llama-index
#
#############################################################################################


#### !!!!! 重要  只运行一次 !!!!!!
create_new_ChromaDB    = False #Existing    # True  创建并储存 新的 ChromaDB   
create_new_qa_datasets = False #Existing    # True  创建并储存 新的 qa_datasets

####### 系统参数总控台 ： #######
splitBytoken=  True     # False by char  用于ic1666
CHUNK_SIZE=  1024  #1666 #char    #334      #374   #430   #490  #2048
OVERLAP_SIZE= 95   #166 #char    #60       #123   #60    #90   #60

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
totalCount=0
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

print(f"\n\n ################################## total records : {totalCount}  {len(metadatas)} " )
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

if model_Type=="azure" or model_Type=="hf":#### 缺省使用 Azure
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


#retriever = vector_index.as_retriever(similarity_top_k=top_k)
query_engine = vector_index.as_query_engine(similarity_top_k=top_k)  


#eval_questions = data_generator.generate_questions_from_nodes(num = 10)  ##### openai/resources/chat/completions.py hardcode  azure integration problem
eval_questions =  [
   'What is the purpose of the Lu0InteractionSpec class in the sample code?',
      'How many core components ?',
      'How to use development tool to create UDTT applications?',
      'Can you summary each runtime component by a short text?',
      'Can you summarize the new features of each UDTT version?',
      'What is the solution architecture and benefits of UDTT?',
      'What is the hardware and software requirement of UDTT?',
       'How does UDTT support multichannel applications?',
       'UDTT known issues and limitations',    
    
        "What is the purpose of multichannel support in UNICOM\u00ae Digital Transformation Toolkit (UDTT\u2122)? How does it benefit the implementation of business functions?",
        "Explain the role of the \"ChannelInitializer\" class in the channel handler configuration. What is its responsibility and in which package is it packaged?",
        "What is the purpose of the \"requestHandler\" field in the given context information?",
        "How does the new release of UNICOM Digital Transformation Toolkit (UDTT) version 10.4 help businesses achieve digital transformation? Provide specific examples of the new features that facilitate this process.",
        "Explain the enhancements made to the UDTT migration tool in version 10.4. How does this tool help customers reduce effort and minimize errors during code and definition changes?",
        "How does the new release of UNICOM Digital Transformation Toolkit (UDTT) help businesses achieve digital transformation? Provide specific examples of the new features introduced in Version 10.3.0.0.",
        "What are the underlying software updates and improvements included in Version 10.3.0.0 of UNICOM Digital Transformation Toolkit? Explain the significance of these updates, particularly in terms of browser support and security vulnerabilities.",
        "How does UDTT Version 10.2.0.0 support the integration of legacy applications with new technologies for digital transformations?",
        "What are the new features included in UDTT Version 10.2.0.0, and how do they enhance the development and usability of the toolkit?",
        "How does UNICOM\u00ae Digital Transformation Toolkit (UDTT\u2122) Version 10.1.0.0 help in integrating legacy applications with new technologies? Provide specific examples of the features included in this release.",
        "What is the purpose of the Initialization Manager in UDTT? How does it allow for customization of components?",
        "How does the ElementFactory in UDTT facilitate Dependency Injection and what is its alternative name?",

        ]
'''
      "Can you provide an example of how UDTT can be used in a non-financial industry?",
      "How does UDTT optimize the processing of transactions?",
       'What are the best practices for UDTT solutions to achieve the best performance results?',
       'How can developers work around the limitation of right-click and left-click in IE browsers?',
       'How can developers validate currency symbols in UDTT?',
       'How does UDTT support microservice architecture?',
       'Can UDTT be used for industries other than banking and finance?',
       'How does UDTT\'s Open API online test work?',
       'What are the benefits of using UDTT for financial institutions?',
       'How does UDTT handle integration with large-scale OLTP systems?',
    'Can you provide an example of how external parameters can be used in UDTT to deliver new functionality without requiring new code?',
    'What are the benefits of using UDTT for reducing application operating costs?',
    'How does UDTT promote externalization of parameters to minimize development effort?',
    ' Can UDTT be used for developing high transaction volume environments?',
    'Can you provide more information on the changes made to NLS organization and file format in version 8.2.0.0 of BTT?',
    ' How do the changes to global variables in version 8.2.1 affect the way users interact with BTT?',
    'What are the best practices for UDTT solutions to achieve the best performance results?',
    'How can developers work around the limitation of right-click and left-click in IE browsers?',
    ' How can developers validate currency symbols in UDTT?',
    ' How does UDTT support the development of mobile banking applications?',
    'What are the benefits of using UDTT for developing multichannel applications?',
    'Can UDTT be used to develop applications for other industries besides banking?',
        "How does the new release of UNICOM Digital Transformation Toolkit (UDTT) version 10.4 help businesses achieve digital transformation? Provide specific examples of the new features that facilitate this process.",
        "Explain the enhancements made to the UDTT migration tool in version 10.4. How does this tool help customers reduce effort and minimize errors during code and definition changes?",
        "How does the new release of UNICOM Digital Transformation Toolkit (UDTT) help businesses achieve digital transformation? Provide specific examples of the new features introduced in Version 10.3.0.0.",
        "What are the underlying software updates and improvements included in Version 10.3.0.0 of UNICOM Digital Transformation Toolkit? Explain the significance of these updates, particularly in terms of browser support and security vulnerabilities.",
        "How does UDTT Version 10.2.0.0 support the integration of legacy applications with new technologies for digital transformations?",
        "What are the new features included in UDTT Version 10.2.0.0, and how do they enhance the development and usability of the toolkit?",
        "How does UNICOM\u00ae Digital Transformation Toolkit (UDTT\u2122) Version 10.1.0.0 help in integrating legacy applications with new technologies? Provide specific examples of the features included in this release.",
        "What is the purpose of the Initialization Manager in UDTT? How does it allow for customization of components?",
        "How does the ElementFactory in UDTT facilitate Dependency Injection and what is its alternative name?",
        "How does the UNICOM\u00ae Digital Transformation Toolkit (UDTT\u2122) facilitate the development of teller applications?",
        "Explain the concept of operation flows and operation steps in the context of the Core component interaction in UDTT\u2122.",
        "How does the concept of operation flow contribute to code reusability in a business operation? Provide an example to support your answer.",
        "In a client/server environment, explain the role and significance of operation contexts on both the client and server sides.",
        "How do data elements enable toolkit entities, processes, and services to manipulate data during runtime?",
        "What is the purpose of formatters in the context of data elements?",
        "What is the purpose of the DataElement class in the toolkit's hierarchy of data elements?",
         "What is the purpose of the \"id\" attribute in the <context> tag? How does it provide access to the data?",
        "Explain the difference between the \"addToDynamicKColl\" attribute being set to false and true in the <context> tag. How does it affect the behavior of the setValueAt method?",
        "What is the difference between typed and non-typed data elements in the context of runtime components? How do they coexist and what is the significance of the getDescriptor method for each type?",
        "Explain the concept of typed data elements and their role in representing business objects. Provide examples of typed data elements and their corresponding business objects.",
        "What is the purpose of a property descriptor in the context of typed data elements?",
        "How are parameters used in property descriptors to describe the type of a data element?",
        "What is the purpose of a default property descriptor in the context of typed data?",
        "How can a default property descriptor be used as a template for creating more specific property descriptors for different types of data?",
        "What is the purpose of the JDBCServicesConnectionManager in the context of the document?",
        "How does the LDAP Access Service enable an application to communicate with an LDAP-compliant directory service?",
        "What is the purpose of the BTTInvoker configuration in the runtime components of the system?",
        "How does the BTTInvoker support different types of invocation, such as POJO, EJB, Web Service, JMS, and synchronized invocation?",
        "What is the purpose of multichannel support in UNICOM\u00ae Digital Transformation Toolkit (UDTT\u2122)? How does it benefit the implementation of business functions?",
        "Explain the role of the \"ChannelInitializer\" class in the channel handler configuration. What is its responsibility and in which package is it packaged?",
        "What is the purpose of the \"requestHandler\" field in the given context information?",
        "In the \"externalizerAccessors\" collection, what are the default configuration values for the \"type\", \"data\", \"context\", \"format\", \"service\", \"operation\", and \"processor\" fields? Can these values be extended or customized?",
        "How do externalizers in the runtime components of the application use external files to initialize defined objects?",
        "Which components in the application presentation layer have an externalizer and a generic XML file containing tags?",
        "What are the two ways in which an externalizer can obtain the full name of the class it is to instantiate in the toolkit?",
        "How do independent definitions provide greater flexibility for implementation classes in the toolkit?"
'''   

faithfulness_azure = FaithfulnessEvaluator(service_context=service_context)  
relevancy_azure = RelevancyEvaluator(service_context=service_context)

  
def evaluate_response_time_and_accuracy(chunk_size):  
    total_response_time = 0  
    total_faithfulness = 0  
    total_relevancy = 0
  
    #service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings_model ,chunk_size=chunk_size )
    #vector_index = VectorStoreIndex.from_documents(  
     #   eval_documents, service_context=service_context  
    #)

    num_questions = len(eval_questions)

    for question in eval_questions:  
        start_time1 = time.time()  
        response_vector = query_engine.query(question)  
        elapsed_time = time.time() - start_time1
        print("\n##### len(response_vector.source_nodes) ",len(response_vector.source_nodes))
        faithfulness_result = faithfulness_azure.evaluate_response(  
            response=response_vector  
        )
        faithfulness_result=faithfulness_result.passing

        relevancy_result = relevancy_azure.evaluate_response(  
            query=question, response=response_vector  
        )
        relevancy_result=relevancy_result.passing

        total_response_time += elapsed_time  
        total_faithfulness += faithfulness_result  
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions  
    average_faithfulness = total_faithfulness / num_questions  
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy

  
for chunk_size in [490]:    #[256, 334, 490, 1024, 2048]  :
  avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size)  
  print(f"Chunk size {CHUNK_SIZE} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
