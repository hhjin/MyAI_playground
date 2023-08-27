from langchain.chains.openai_functions.openapi import get_openapi_chain
import langchain
langchain.verbose=True


chain = get_openapi_chain("https://api.speak.com/openapi.yaml", verbose=True)

import json

# Insertion
data = {"name": "John", "age": 30, "city": "New York"}
print(json.dumps(data, indent=4, ensure_ascii=False))

 
#词语解释plugin API
output= chain.run("How would you say \'Pydantic\' in Chinese")
 
print("###### explanation:\n" ,output.get("explanation") )
print("\n###### extra_response_instructions:\n" ,output.get("extra_response_instructions") )
 


#chain = get_openapi_chain("https://gist.githubusercontent.com/roaldnefs/053e505b2b7a807290908fe9aa3e1f00/raw/0a212622ebfef501163f91e23803552411ed00e4/openapi.yaml" , verbose=True)
#print( chain.run("What's the today's comic?") )