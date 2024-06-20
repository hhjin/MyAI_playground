
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY_openai")
openai.api_type="openai"
openai.api_key ="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJwd2RfYXV0aF90aW1lIjoxNzE3NDg2NDc5ODg5LCJzZXNzaW9uX2lkIjoiXzVTb0pYQ056XzZxN1h0NXhqVzNXai1pdkhzQ1VoMU4iLCJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJoaGppbmhoQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS9hdXRoIjp7InBvaWQiOiJvcmctcHdsNDVhQ2FHRmxaaWhNSGZLYWExMnhFIiwidXNlcl9pZCI6InVzZXItUlZyWmYyMDJOSUVPclIwRFZGMTI2RFNyIn0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExMTI4NTcwMTIwOTM5ODY0ODg5NSIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE3MTc0ODY0ODAsImV4cCI6MTcxODM1MDQ4MCwic2NvcGUiOiJvcGVuaWQgcHJvZmlsZSBlbWFpbCBtb2RlbC5yZWFkIG1vZGVsLnJlcXVlc3Qgb3JnYW5pemF0aW9uLnJlYWQgb3JnYW5pemF0aW9uLndyaXRlIG9mZmxpbmVfYWNjZXNzIiwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEcifQ.PV2m55GuR11k4A4pzYKwp2spBYQhj87GxjcyMhUzW8b4t__5bICBaE_gAgXcXOpFvHgVw01VYwkALLe4gFAkpQI1mJS6IWRHd7JZzIOBrylyKI1UHDz0hcN05UXCyX5q2Q3sDX0SJS912OuImN1jNLHi6R9qZ5ViR7fDhEr2ZD_A23VRAXy9cHPqjZSnS2D4ayRDH2qURPj-t-4IxP-w2c_AV4elqAumcnEDlFd7ZN3mzcYnAF06rghznIRnj4Znk-VGUkmyZMykYFH1axcN41MBnpjd4iJKRfaGFfB2H2vTZ0f4DstB4WBmITDwvhNwnGf0Z0VqDIsRIotHmauGzw"
openai.api_base ="http://10.1.44.101:80/v1"
openai.api_base ="http://20.55.112.76:80/v1"

messages = []
system_msg = "You are a helpful assistant who always answer in Chinese."
messages.append({"role": "system", "content": system_msg})
message=""
while message != "quit()":
    message = input("\n You    : ")

    import time
    start_time = time.time()      
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-16k",
        model="gpt-4o",
        messages=messages,
        max_tokens=100,

    )
    replay = response["choices"][0]["message"]["content"]
    messages.append({"role": "system", "content": replay})
    print(' GPT : ' + replay )
     
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")


    ''' # test Embedding
    response=openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )
    print(" embedding usage:",response.usage.total_tokens)
    print("embedding data:",response.data[0].embedding)
    '''

