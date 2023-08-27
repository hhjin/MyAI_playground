#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai


''' 
#######  2022-12-01 version is for completion API
    of deployment engin :  gpt35turbo  gpt35turbo-16k    text-embedding-ada-002

openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY_azure")


for i in range(500):
    promptInput = input("\n Input your mesage for azure CHAT-GPT :")

    response = openai.Completion.create(
        engine="gpt35turbo",
        #prompt="Generate a summary of the below conversation in the following format:\nCustomer problem:\nOutcome of the conversation:\nAction items for follow-up:\nCustomer budget:\nDeparture city:\nDestination city:\n\nConversation:\nUser: Hi there, I’m off between August 25 and September 11. I saved up 4000 for a nice trip. If I flew out from San Francisco, what are your suggestions for where I can go?\nAgent: For that budget you could travel to cities in the US, Mexico, Brazil, Italy or Japan. Any preferences?\nUser: Excellent, I’ve always wanted to see Japan. What kind of hotel can I expect?\nAgent: Great, let me check what I have. First, can I just confirm with you that this is a trip for one adult?\nUser: Yes it is\nAgent: Great, thank you, In that case I can offer you 15 days at HOTEL Sugoi, a 3 star hotel close to a Palace. You would be staying there between August 25th and September 7th. They offer free wifi and have an excellent guest rating of 8.49/10. The entire package costs 2024.25USD. Should I book this for you?\nUser: That sounds really good actually. Please book me at Sugoi.\nAgent: I can do that for you! Can I help you with anything else today?\nUser: No, thanks! Please just send me the itinerary to my email soon.\n\nSummary:",
        prompt=promptInput ,
        temperature=0,
       
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )      

    print(response.choices[0].text)

    response = openai.Embedding.create(
        input=promptInput,
        engine="text-embedding-ada-002"

    )
    embeddings = response['data'][0]['embedding']

    print("embeddings :") 
    print(embeddings[0])
'''

openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("OPENAI_API_KEY_azure")


for i in range(500):
    promptInput = input("\n Input your mesage for azure CHAT-GPT :")

    response = openai.Completion.create(
        engine="gpt35turbo",
        #prompt="Generate a summary of the below conversation in the following format:\nCustomer problem:\nOutcome of the conversation:\nAction items for follow-up:\nCustomer budget:\nDeparture city:\nDestination city:\n\nConversation:\nUser: Hi there, I’m off between August 25 and September 11. I saved up 4000 for a nice trip. If I flew out from San Francisco, what are your suggestions for where I can go?\nAgent: For that budget you could travel to cities in the US, Mexico, Brazil, Italy or Japan. Any preferences?\nUser: Excellent, I’ve always wanted to see Japan. What kind of hotel can I expect?\nAgent: Great, let me check what I have. First, can I just confirm with you that this is a trip for one adult?\nUser: Yes it is\nAgent: Great, thank you, In that case I can offer you 15 days at HOTEL Sugoi, a 3 star hotel close to a Palace. You would be staying there between August 25th and September 7th. They offer free wifi and have an excellent guest rating of 8.49/10. The entire package costs 2024.25USD. Should I book this for you?\nUser: That sounds really good actually. Please book me at Sugoi.\nAgent: I can do that for you! Can I help you with anything else today?\nUser: No, thanks! Please just send me the itinerary to my email soon.\n\nSummary:",
        prompt=promptInput ,
        temperature=0,
       
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )      

    print("\n\n\n #########$$$$$$$$$$$$$  response .... :",response.choices[0].text)
    
    #openai.api_version = "2022-12-01"
    response = openai.Embedding.create(
        input=promptInput,
        engine="embedding2",
        # model="text-embedding-ada-002",

    )
    embeddings = response['data'][0]['embedding']

    print("embeddings :") 
    print(embeddings[0])

