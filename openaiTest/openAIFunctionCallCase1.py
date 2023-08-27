import openai
def classify(input_string:str)->str:
    functions= [ {

        "name":"print_sentiment",

        "description":"A function that prints the given sentiment", 
        
        "parameters": {
            "type":"object", 
            "properties": {
                "sentiment": {
                    "type":"string",
                    "enum":["positive","negative","neutral"], 
                    "description":"The sentiment.",
                },
            },
            "required":["sentiment"],
        }
    }]

    messages= [{"role":"user","content":input_string}] 
    
    response=openai.ChatCompletion.create(

        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call={"name":"print_sentiment"},
    )

    function_call=response.choices [0].message ["function_call"] 
    print (function_call)
    argument= json.Loads(function_call["arguments"]) 
    print (argument)
    return argument


classify("I love icecream!")
 