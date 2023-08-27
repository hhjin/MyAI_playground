
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.environ.get('OPENAI_API_KEY_azure')

messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})
message=""
while message != "quit()":
    message = input("\n You    : ")
                
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        engine="gpt35turbo",
        messages=messages
    )
    replay = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": replay})
    print('\n\n#######\n GPT3.5 Azure: ' + replay )

    
    response=openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )
    
    print(" embedding usage:",response.usage.total_tokens)
    
    # print("embedding data:",response.data[0].embedding)
    

