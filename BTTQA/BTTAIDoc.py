from langchain.llms import OpenAI
from typing import Any, Dict, List, Optional
import langchain
from langchain import PromptTemplate ,LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain ,RetrievalQA
from langchain.llms import Cohere
import sys
import time
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
from langchain.chains import create_qa_with_sources_chain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

 ###########################################################################
#   一些初期的 基本 langchain 功能的研究 ,基于完成llm（也能换成chatLLM,回答效果更好） 以实现 BTTAIDoc 
#   bttqa_prompt_chain :  用简单的的prompt和 基础 chain 实现 BTTAIDoc的基本功能 
#   bttqa_qa_chain  : 支持chain type ，
#   RetrievalQA chain 封装程度最高，但用户可控制最少， 而且verbose在cursor console没输出，
# 
#   综合评估，完成LLM推荐使用load_qa_chain
#
#  LLama  模型试用 2023/07/25 
#   7B 无法完成问答
#   13B Llama -2 和 vicuna 13b 能完成Q/A ，但质量和ChatGPT无法比
#   速度很慢，10 token/second
#   
#  llama.cpp  参数不用调用缺省， 调也没用，特别是设置n_batch好像还会变慢
#  
#  句向量模型也很不好用，查询质量很差
############################################################################


# QA by  PromptTemplate chain (completion API) 
def bttqa_prompt_chain(llm, store, query, PROMPT,  topk=5):
 
    print(f"\n\n###################   bttqa_prompt_chain,   store: {store} \n\n########### LLM : {llm}\n")
    start_time = time.time()
    docs = store.similarity_search(query, k=topk)
    end_time = time.time()
    run_time = end_time - start_time
    print("程序similarity_search运行时间：%.2f 秒" % run_time)

    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    inputs= {"context": contexs, "question":query.strip()}

    print(f"\n\n\n################### bttqa_prompt_chain, prompt : \n {PROMPT}   \n\n################## contexs : \n {contexs}")

    chain = LLMChain(llm=llm, prompt=PROMPT)
    print(f"\n######################### Answer for query :  {query} \n  ")
    response = chain.run(inputs)
    if (isinstance(llm, Cohere) or llm.streaming ==False ) :
      print(response)
    return response


# QA by  qa_chian   (completion API) 
#  chain_type : (['stuff', 'map_reduce', 'refine', 'map_rerank'])
def bttqa_qa_chain(llm, store, query , PROMPT: Optional=None,chain_type="stuff" ,  topk=5 ):
    print(f"\n\n###################   bttqa_qa_chain,   store: {store} \n\n########### LLM : {llm}\n")
    
    docs = myQAKit.similarity_search(store,query, k=topk)

    if (chain_type=="map_reduce" or chain_type=="map_rerank" ) :
      llm.streaming=False
    if PROMPT==None :
        chain = load_qa_chain( llm=llm, chain_type=chain_type )
    else :
        chain = load_qa_chain( llm=llm, chain_type=chain_type , prompt=PROMPT )
        #chain = load_qa_with_sources_chain( llm=llm, chain_type="map_reduce", return_map_steps=True, question_prompt=PROMPT, combine_prompt=COMBINE_PROMPT)

    #response = chain.run(input_documents=docs, question=query.strip())
    response=chain({"input_documents": docs, "question": query}, return_only_outputs=True)


    print(f"\n\n\n######################### Answer for query :  {query} \n\n ")


    if (isinstance(llm, Cohere) or llm.streaming ==False ) :
      print(response)
    return response
  


#############     main start   #############################  
prompt_template_llama2 ="""SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. When given CONTEXT you answer questions using only that information. After your answer, you act as an user instead of assistant by asking three specific insightful further questions about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. You always format your output in markdown. You must only use the CONTEXT provided to form your answer, but avoid copying word-for-word from the context text. If you don't know the answer to a question, please don't share false information.
-------------
Context: {context}
-------------
USER Question: {question} You remember to provide further questions about the provided context.
Assistant:"""
PROMPT_llama2 = PromptTemplate(   template=prompt_template_llama2, input_variables=["context", "question"]  )

prompt_template = """You are a helpful assistant on IT area. When given CONTEXT you answer questions using only that information. You always format your output in markdown. After your answer, you act as an user instead of assistant by asking three specific insightful further questions 
  about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. If you are unsure and the answer is not explicitly written in the CONTEXT provided, you say
  \"Sorry, I don't know how to help with that.\". Use the text provided to form your answer, but avoid copying word-for-word from the context text. 
  You try to use your own words when possible. If the CONTEXT includes SOURCE URLs you include them in your answer at the text where is relevant with the SOURCE URLs. Always include all of the relevant source urls from the CONTEXT, Never make up URLs.\n
-------------
    Context: {context}
-------------
    Question: {question}"""
  
PROMPT = PromptTemplate(   template=prompt_template, input_variables=["context", "question"]  )

myQAKit=QA_Toolkit("./Chroma_DB_300.2300")
storeOpenAI_Supabase=myQAKit.get_dbstore_supabase()
storeOpenAI_Chroma=myQAKit.get_dbstore_openai()

llmAzure500=myQAKit.get_llm_azure( max_tokens=500)
llmAzure3000=myQAKit.get_llm_azure( max_tokens=3000)
llmAzureChat=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=3300, temperature=0,)
llmcohere=myQAKit.get_llm_cohere(temperature=0.9)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llamaCpp = LlamaCpp(
    model_path="/Users/henryking/llama.cpp/models/llama-2-7b.ggmlv3.q5_K_M.bin",
    #model_path="/Users/henryking/llama.cpp/models/ggml-vic7b-q5_1.bin",
    input={"temperature": 0.5, "max_length": 3000, "top_p": 1},
    n_ctx=2900,
    callback_manager=callback_manager,
    verbose=True,
)



# OpenAI Q/A ，封装程度最高，但用户可控制最少， 而且verbose在cursor console没输出，综合评估，完成LLM推荐使用load_qa_chain
chain_type_kwargs = {"prompt": PROMPT}
qaRetrieval_chain = RetrievalQA.from_chain_type(
    llm=llmcohere, 
    chain_type="stuff", 
    retriever= storeOpenAI_Chroma.as_retriever(), 
    verbose=True, 
    chain_type_kwargs=chain_type_kwargs
)
#langchain.debug=True
langchain.verbose = True
while True:
        query =input("\n\n####### Question for BTT: ")
        if query==" ":    # default query for space input        
          query="What is new in each UDTT version?"
        else :
          if query=="":
                break
        # query=query+" Answer in Chinese with mark down format with empty lines. Don't repeat content and copy context. You end response after you asked three further questions"
        
        #response=bttqa_prompt_chain(llmAzureChat, storeOpenAI_Chroma, query, PROMPT)

        # test llamaCpp
        response=bttqa_qa_chain (llamaCpp, storeOpenAI_Chroma, query, PROMPT_llama2, "stuff",1)

        #response = qaRetrieval_chain.run(query)
 


 ###########################################################################
#   一些初期的 基本 langchain 功能的研究 ,基于完成llm（也能换成chatLLM,回答效果更好） 以实现 BTTAIDoc 
#   bttqa_prompt_chain :  用简单的的prompt和 基础 chain 实现 BTTAIDoc的基本功能 
#   bttqa_qa_chain  : 支持chain type ，
#   RetrievalQA chain 封装程度最高，但用户可控制最少， 而且verbose在cursor console没输出，
# 
#   综合评估，完成LLM推荐使用load_qa_chain

##  LLama  模型试用 2023/07/25 
#   7B 无法完成问答
#   13B Llama -2 和 vicuna 13b 能完成Q/A ，但质量和ChatGPT无法比
#   速度很慢，10 token/second
#   
#  llama.cpp  参数不用调用缺省， 调也没用，特别是设置n_batch好像还会变慢
#  句向量模型也很不好用，查询质量很差
############################################################################
