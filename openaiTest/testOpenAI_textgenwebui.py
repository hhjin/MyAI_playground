
import openai
import os

# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples
# py server.py  --extensions openai
openai.api_type = "openai"
openai.api_base = "http://127.0.0.1:5001/v1"
openai.api_key = '########****os.getenv("OPENAI_API_KEY")'
completion=None;

print(openai.api_key )
for i in range(500):
  promptInput = input("\n Input your mesage for CHAT-GPT :")

 
  completion=openai.Completion.create(
          engine="gpt35turbo",
          model="text-davinci-003",
          prompt=promptInput ,
          temperature=0.98,
          max_tokens=350,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )
    
  print(completion.choices[0].text)
  
  # embedding by HuggingFace model: sentence-transformers/all-mpnet-base-v2 for embeddings. This produces 768 dimensional embeddings (the same as the text-davinci-002 embeddings),
  response = openai.Embedding.create(
        input=promptInput,
        engine="embedding2",
        # model="text-embedding-ada-002",

    )
  embeddings = response['data'][0]['embedding']

  print("embeddings :") 
  print(embeddings[0])