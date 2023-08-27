from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import Cohere
import os

chat = ChatOpenAI(temperature=0.0 )

chatAzure = AzureChatOpenAI(
    openai_api_base= "https://bttaidoc.openai.azure.com/",
    openai_api_version="2023-03-15-preview",
    deployment_name='gpt35turbo',
    openai_api_key=os.getenv("OPENAI_API_KEY_azure"),
    openai_api_type="azure",
)

print(chatAzure)

template = """Use the following pieces of context to answer the question at the end. Use three sentences maximum. 
{context}
Question: {question}
Answer: Think step by step """

question= "What is the summary of the text below that captures its main idea?"
context="\n\ncontext: At Microsoft, we have been on a quest to advance AI beyond existing techniques, by taking a more holistic, human-centric approach to learning and understanding. As Chief Technology Officer of Azure AI Cognitive Services, I have been working with a team of amazing scientists and engineers to turn this quest into a reality. In my role, I enjoy a unique perspective in viewing the relationship among three attributes of human cognition: monolingual text (X), audio or visual sensory signals, (Y) and multilingual (Z). At the intersection of all three, there’s magic—what we call XYZ-code as illustrated in Figure 1—a joint representation to create more powerful AI that can speak, hear, see, and understand humans better. We believe XYZ-code will enable us to fulfill our long-term vision: cross-domain transfer learning, spanning modalities and languages. The goal is to have pre-trained models that can jointly learn representations to support a broad range of downstream AI tasks, much in the way humans do today. Over the past five years, we have achieved human performance on benchmarks in conversational speech recognition, machine translation, conversational question answering, machine reading comprehension, and image captioning. These five breakthroughs provided us with strong signals toward our more ambitious aspiration to produce a leap in AI capabilities, achieving multi-sensory and multilingual learning that is closer in line with how humans learn and understand. I believe the joint XYZ-code is a foundational component of this aspiration, if grounded with external knowledge sources in the downstream AI tasks."


# --------------  openAI chat api call   ----------------------------  

prompt_template = ChatPromptTemplate.from_template(template)
customer_messages = prompt_template.format_messages(
                    question=question,
                    context=context)
print(type(customer_messages))
print(type(customer_messages[0]))
print(    (customer_messages[0]))

customer_response = chatAzure(customer_messages)

print("\n\n\n\n ############ chatAzure : ",customer_response.content)



# --------------  llama langchain completion call  ------------------------

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager



llmCohere = Cohere(cohere_api_key="fggqIlX01EF3SdhYFdeYxaGBx5CcpIEUDRjEHbcS",
    temperature= 0,
    max_tokens=800,
)

prompt = PromptTemplate(template=template, input_variables=["question","context"])
context_and_question_chain={'question': question,
 'context': context}

llm_chain_cohere= LLMChain(prompt=prompt, llm=llmCohere)
print("\n\n\n\n ############ llmCohere : ",llm_chain_cohere.run(context_and_question_chain))

# Make sure the model path is correct for your system!
llamaCpp = LlamaCpp(
                model_path="/Users/henryking/llama.cpp/models/ggml-vic13b-q5_1.bin",
                callback_manager=callback_manager,
                verbose=True,
                n_threads=6,
                n_ctx=2048,
                use_mlock=True) 
'''
llamaCpp = LlamaCpp(
    model_path="/Users/henryking/llama.cpp/models/llama-2-7b.ggmlv3.q5_K_M.bin",
    
    input={"temperature": 0.5, "max_length": 3000, "top_p": 1},
    n_ctx=2900,
    callback_manager=callback_manager,
    verbose=True,
)
'''
llm_chain_llamacpp = LLMChain(prompt=prompt, llm=llamaCpp)

print("\n\n\n\n ############ LlamaCpp : ",llm_chain_llamacpp.run(context_and_question_chain))