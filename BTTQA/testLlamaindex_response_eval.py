import nest_asyncio

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
import openai
from llama_index.llms import AzureOpenAI,OpenAI
from llama_index.embeddings import AzureOpenAIEmbedding ,OpenAIEmbedding
try:
    OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
    try:
        OPENAI_API_BASE= os.environ["OPENAI_API_BASE"]
    except KeyError as e:
        OPENAI_API_BASE=os.environ["AZURE_OPENAI_ENDPOINT"]
    OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]
    cache_folder= os.environ["TRANSFORMERS_CACHE"]
except KeyError as e:
	raise Exception(f"Please set environment variables for {e.args[0]}")
	...
# 使用Azure OpenAI Service
azure_lm = AzureOpenAI(engine="gpt35turbo-16k",  temperature=0.0, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
azure_embeddings=AzureOpenAIEmbedding(model="embedding2", embed_batch_size=100000, azure_endpoint=OPENAI_API_BASE, api_key=OPENAI_API_KEY, api_version=OPENAI_API_VERSION)
# OpenAI 支持需要 .zprofile_openai
#llm = OpenAI(temperature=0, model="gpt-3.5-turbo")  
#gpt4 = OpenAI(temperature=0, model="gpt-4")
#openai_embeddings=OpenAIEmbedding(model="text-embedding-3-small")  #text-embedding-3-large  text-embedding-3-small

openai.api_key = OPENAI_API_KEY

reader = SimpleDirectoryReader("./paul_graham_eval/pdf/")  
documents = reader.load_data()

eval_documents = documents[:7]  
data_generator = DatasetGenerator.from_documents(eval_documents)  
#eval_questions = data_generator.generate_questions_from_nodes(num = 10)  ##### openai/resources/chat/completions.py hardcode  azure integration problem
eval_questions =  [
         #"How did the author's experience with writing short stories in their early years contribute to their development as a writer? Provide examples from the context information to support your answer.",
         #"Describe the author's initial encounter with programming on the IBM 1401. What challenges did they face and how did this experience shape their understanding of programming? Use specific details from the context information to support your response.",
       #"In the context of the 1401 computer, what were the limitations of programming at that time? How did the lack of input options affect the author's ability to write programs?",
        "How does the Freight service revolutionize the logistics industry?",
        ]

  
service_context_azure = ServiceContext.from_defaults(llm=azure_lm)

faithfulness_azure = FaithfulnessEvaluator(service_context=service_context_azure)  
relevancy_azure = RelevancyEvaluator(service_context=service_context_azure)

def evaluate_response_time_and_accuracy(chunk_size):  
    total_response_time = 0  
    total_faithfulness = 0  
    total_relevancy = 0
 
    #service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size)  
    service_context = ServiceContext.from_defaults(llm=azure_lm, embed_model=azure_embeddings ,chunk_size=chunk_size )
    vector_index = VectorStoreIndex.from_documents(  
        eval_documents, service_context=service_context  
    )

    query_engine = vector_index.as_query_engine(similarity_top_k=2)  
    num_questions = len(eval_questions)

    for question in eval_questions:  
        start_time = time.time()  
        response_vector = query_engine.query(question)  
        elapsed_time = time.time() - start_time
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

  
for chunk_size in [512]:    #[128, 256, 512, 1024, 2048]  :
  avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size)  
  print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
