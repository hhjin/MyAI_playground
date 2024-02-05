#import faiss
#from langchain.vectorstores import FAISS
from typing import Any, Dict, List, Optional
import time
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from supabase.client import Client, create_client
from langchain.vectorstores import SupabaseVectorStore
from langchain.llms import OpenAI ,AzureOpenAI
from langchain.llms import Cohere
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from BCEmbedding import RerankerModel    
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QA_Toolkit():

    def __init__(self,
        ChromaDB_path: Optional[str] = None,):
        self.ChromaDB_path=ChromaDB_path

    def get_chromadb_path(self) :
        return self.ChromaDB_path
        
    def get_dbstore_supabase(self,
        table_name="documents_langchain",
        query_name="match_documents_langchain",
        ) :
        start_time = time.time()
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        langchain_embeddings_azureopenai = OpenAIEmbeddings(deployment="embedding2")
        vector_store = SupabaseVectorStore( embedding=langchain_embeddings_azureopenai, client=supabase, table_name=table_name , query_name=query_name)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load Subpabase  {table_name} , {supabase_url} , {query_name} , cost time: %.2f 秒\n" % run_time)
        return vector_store

    def get_dbstore_openai(self) :
        start_time = time.time()
        BTTDocDB_path_OpenAI=self.ChromaDB_path+"/OpenAI"
        #langchain_embeddings_azureopenai = OpenAIEmbeddings(deployment="embedding2")
        langchain_embeddings_azureopenai = AzureOpenAIEmbeddings(deployment="embedding2")
        #langchain_embeddings_azureopenai = OpenAIEmbeddings()
        
        storeOpenAI=Chroma(persist_directory=BTTDocDB_path_OpenAI , embedding_function=langchain_embeddings_azureopenai)
        #the _embedding_function should be same when the store is created
        #storeOpenAI._embedding_function=langchain_embeddings_azureopenai
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load ChromaDB  {BTTDocDB_path_OpenAI} , cost time: %.2f 秒\n" % run_time)
        time.sleep(0.8)  # Azure rate limit aviod
        return storeOpenAI
    
    def get_dbstore_cohere(self, model="large") :
        start_time = time.time()
        BTTDocDB_path_Cohere=self.ChromaDB_path+"/Cohere"
        langchain_embeddings_Cohere  = CohereEmbeddings(cohere_api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS" ,model=model)
        storeCohere=Chroma(persist_directory=BTTDocDB_path_Cohere , embedding_function=langchain_embeddings_Cohere)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load ChromaDB  {BTTDocDB_path_Cohere} , cost time: %.2f 秒\n" % run_time)
        return storeCohere

    def get_dbstore_sentence(self) :
        start_time = time.time()
        BTTDocDB_path_SETF=self.ChromaDB_path+"/SETF"
        storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings())
        cache_folder= os.environ["TRANSFORMERS_CACHE"]
        model_name="all-MiniLM-L6-v2"
        if cache_folder :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name,cache_folder=cache_folder))
        else :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name))

        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load ChromaDB  {BTTDocDB_path_SETF} , cost time: %.2f 秒\n" % run_time)
        return storeSETF

    def get_dbstore_bce(self) :
        start_time = time.time()
        BTTDocDB_path_SETF=self.ChromaDB_path+"/BCE"
         
        cache_folder= os.environ["TRANSFORMERS_CACHE"]
        model_name="maidalun1020/bce-embedding-base_v1"
        if cache_folder :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name,cache_folder=cache_folder))
        else :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name))
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load ChromaDB  {BTTDocDB_path_SETF} , cost time: %.2f 秒\n" % run_time)
        return storeSETF
    
    def get_dbstore_BGE(self) :
        #print(f"\n....load ChromaDB  start {self.ChromaDB_path} ")
        start_time = time.time()
        
        BTTDocDB_path_SETF=self.ChromaDB_path+"/BGE_large_en_v1.5"
        try:
            cache_folder= os.environ["TRANSFORMERS_CACHE"]
        except KeyError as e:
            cache_folder=None

        model_name="BAAI/bge-large-en-v1.5"
        if cache_folder :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name,cache_folder=cache_folder))
        else :
            storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name=model_name))
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....load ChromaDB  {BTTDocDB_path_SETF} , cost time: %.2f 秒\n" % run_time)
        return storeSETF

    def get_llm_azure(self,
        deployment_name :Optional[str]="gpt35turbo",
        temperature=0,
        streaming=True,
        max_tokens=740,
        model_name="text-davinci-003", ###### Not useful. The deployment binding model is used.
        callbacks=[StreamingStdOutCallbackHandler()],
     ) :

        llm_azure= AzureOpenAI(
             
            deployment_name=deployment_name,
            model_name=model_name,  
            temperature= temperature,
            max_tokens=max_tokens,
            streaming=streaming,  # Cannot stream results with multiple prompts.
            callbacks=callbacks
        )
        return llm_azure
    
     
    def get_chat_azure(self, 
        deployment_name :Optional[str]="gpt35turbo",  # gpt35turbo-16k   , gpt35turbo
        openai_api_version :Optional[str]="2023-03-15-preview",
        temperature=0,
        streaming=True,
        max_tokens=3000,
        callbacks=[StreamingStdOutCallbackHandler()],
     ) :

        chatAzure = AzureChatOpenAI( 
            openai_api_version=openai_api_version,
            deployment_name=deployment_name,
            temperature= temperature,
            max_tokens=max_tokens,
            streaming=streaming,  # Cannot stream results with multiple prompts.
            callbacks=callbacks ,
        )
        return chatAzure
    
    def get_llm_openai(self,
        model_name :Optional[str]="text-davinci-003", 
        temperature=0,
        streaming=True,
        max_tokens=800,
        callbacks=[StreamingStdOutCallbackHandler()],
     ) :
        return OpenAI(
             
            model_name=model_name,
            temperature= temperature,
            max_tokens=max_tokens,
            streaming=streaming,  # Cannot stream results with multiple prompts.
            callbacks=callbacks ,
        )
    
    # Cohere 不支持Chatchain ,&& Unicom Office network
    def get_llm_cohere(self,
            temperature=0,
            max_tokens=500,
        ) :
        llmCohere = Cohere(cohere_api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS",
            temperature= temperature,
            max_tokens=max_tokens ,
        )
        return llmCohere

    def  printMatchedDocsWithScore(self, result_list) :
        # 打印输出
        for document, score in result_list:
            print("\n\n\n #######¢§∞∞∞∞§§§§§§££££™¢∞∞∞######   文档内容 :  ######¢§∞∞∞∞§§§§§§££££™¢∞∞∞######\n\n", 
            document.page_content, "\n\n ####### 相似度得分: ", score,"\n\n ############ Source:\n     ",document.metadata.get("source"))
            #print(f"Content: {document.page_content}, Metadata: {document.metadata}, Score: {score}")

    def  printMatchedDocs(self, result_list) :
        # 打印输出
        for document in result_list:
            print("\n\n\n #######¢§∞∞∞∞§§§§§§££££™¢∞∞∞######   文档内容 :  ######¢§∞∞∞∞§§§§§§££££™¢∞∞∞######\n\n", 
            document.page_content, "\n\n ############ Source:\n     ",document.metadata.get("source"))

    '''
    def loadFAISStore(self, indexFile , pklName) -> FAISS :
        print ("load index file : "+indexFile)
        index = faiss.read_index(indexFile)

        print ("load pkl store file : "+pklName)
        with open(pklName, "rb") as f:
            store = pickle.load(f)
        
        store.index = index
        return store
    '''

    def similarity_search_with_score(self,  store , query,   k=5) :
        start_time = time.time()
        matchedDocs=store.similarity_search_with_score(query , k)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....{store.__class__.__name__} -- similarity_search_with_score  cost time: %.2f 秒\n" % run_time)
        self.printMatchedDocsWithScore (matchedDocs)
        return matchedDocs

    def similarity_search( self, store ,query, k=5) :
        start_time = time.time()
        matchedDocs=store.similarity_search(query, k)
        end_time = time.time()
        run_time = end_time - start_time

        print(f"\n....{store.__class__.__name__} -- similarity_search  cost time: %.2f 秒\n" % run_time)
        self.printMatchedDocs(matchedDocs)
        return matchedDocs
        
    def max_marginal_relevance_search( self, store ,query, k=5) :
        start_time = time.time()
        matchedDocs=store.max_marginal_relevance_search(query, k)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....{store.__class__.__name__} -- max_marginal_relevance_search  cost time: %.2f 秒\n" % run_time)
        self.printMatchedDocs(matchedDocs)
        return matchedDocs
    

    def reRanker_BCEbase(self,query, docs,topk=10) :

        start_time = time.time()
        # your query and corresponding passages
        query = query.strip()
        passages = [ doc.page_content.strip() for doc in docs]

        #  # 为了快速本地加载,需要修改RerankerModel 源码,AutoTokenizer 添加 local_files_only=True
        model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1",local_files_only=True)


        #  在 RerankerModel.rerank 方法中,当“通道”很长时,我们提供了一种先进的预处理程序,用于生产 sentence_pairs 。
        rerank_results = model.rerank(query, passages)
        print(rerank_results)
        rerank_scores=rerank_results.get("rerank_scores")
        rerank_ids= rerank_results.get("rerank_ids")
        rerank_passages=rerank_results.get("rerank_passages")
        for i, doc in enumerate(rerank_passages):
            print(f"\n\n ###### Rerank score: {rerank_scores[i]}   ### idx: {rerank_ids[i]}    ###### \n   {doc[:200]}")   

        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....reRanker_BCEbase cost time: %.2f 秒\n" % run_time)  
        return rerank_passages[:topk]
    


    def reRanker_BGE_large(self,query, docs, topk=10) :
        rerank_passages=[]
        start_time = time.time()
        # your query and corresponding passages
        query = query.strip()
        pairs = [ [query,  doc.page_content.strip()]  for doc in docs]
        
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large' ,local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large',local_files_only=True)
 
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            # all are tensor for scores sorted_indexes sorted_scores
            sorted_indexes = np.argsort(-scores)
            sorted_scores = scores[sorted_indexes]  
            for i, score in enumerate(sorted_scores):
                original_idx= sorted_indexes[i].item()
                origianl_passage=pairs[original_idx][1]
                rerank_passages.append(origianl_passage)
                print(f"\n\n ###### Rerank score: {scores[original_idx].item()}   ### idx: {original_idx}   ###### \n {origianl_passage[:200]} ")  
               
        end_time = time.time()
        run_time = end_time - start_time
        print(f"\n....reRanker_BGE_large cost time: %.2f 秒\n" % run_time)

        return rerank_passages[:topk]
    
'''
import numpy as np
scores = np.array([0.1, 0.8, 0.7, 0.9])
sorted_indexes = np.argsort(-scores)
print(sorted_indexes)
# [3 1 2 0]
sorted_scores = scores[sorted_indexes]  
print(sorted_scores)
# [0.9 0.8 0.7 0.1]
'''

    

    