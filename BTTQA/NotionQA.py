import sys
from langchain.chains import RetrievalQAWithSourcesChain ,RetrievalQA
from langchain.chains.question_answering import load_qa_chain

sys.path.append('./utils')
from mylangchainutils import QA_Toolkit
 
# read Chroma DB as vector store
myQAKit=QA_Toolkit("./Chorma_NotionDB")
cohereNLS_embdStore=myQAKit.get_dbstore_cohere( model="multilingual-22-12")

llmAzure=myQAKit.get_llm_azure(max_tokens=100, model_name="text-ada-001",)
llmcohere=myQAKit.get_llm_cohere(temperature=0.9)
llmopenai=myQAKit.get_llm_openai(temperature=0.9)

while True:
        query =input("\n\n####### Query for notionDB: ")
        if query=="":
                break

        docs = myQAKit.similarity_search(cohereNLS_embdStore,query, k=6)

        print("\n\n\n\n################### load_qa_chain : "+query)
        chain = load_qa_chain( llmcohere, chain_type="stuff")
        chain.run(input_documents=docs, question=query)


        print("\n\n\n\n################### RetrievalQA -llmcohere : "+query)
        qa_stuff = RetrievalQA.from_chain_type(
                llm=llmcohere, 
                chain_type="stuff", 
                retriever= cohereNLS_embdStore.as_retriever(), 
                verbose=False, 
        )
        qa_stuff.run(query)
   

