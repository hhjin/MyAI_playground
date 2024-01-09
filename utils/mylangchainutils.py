import faiss
from langchain.vectorstores import FAISS
from typing import Any, Dict, List, Optional
import pickle
import time
import os
from openai import  InvalidRequestError 
from IPython.display import display, Markdown
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
from langchain.embeddings import OpenAIEmbeddings
    

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
        langchain_embeddings_azureopenai = OpenAIEmbeddings(deployment="embedding2")
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
        storeSETF=Chroma(persist_directory=BTTDocDB_path_SETF , embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
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

    def loadFAISStore(self, indexFile , pklName) -> FAISS :
        print ("load index file : "+indexFile)
        index = faiss.read_index(indexFile)

        print ("load pkl store file : "+pklName)
        with open(pklName, "rb") as f:
            store = pickle.load(f)
        
        store.index = index
        return store

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

    

    