import torch
from transformers import pipeline
import time

###########################################################################
#
#  和 faster_whisper 项目没 关系， 纯的support by transformers huggingface  
# https://github.com/Vaibhavs10/insanely-fast-whisper/tree/main


pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2" , #  "openai/whisper-large-v2",  "openai/whisper-tiny"
                torch_dtype=torch.float16,
                device="cuda:0")  #   device="cuda:0")

pipe.model = pipe.model.to_bettertransformer()

# 记录程序开始时间
start_time = time.time()

outputs = pipe("d:/Downloads/sam_altman_lex_podcast_367.flac", # m4a not supported
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

print(outputs["text"])

# 记录程序结束时间
end_time = time.time()
# 计算运行时间
run_time = end_time - start_time
# 打印运行时间
print("程序运行时间：%.2f 秒" % run_time)


# 只在高级 GPU 上入A100 才支持 flash_attention
'''
import torch
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition",
               "openai/whisper-tiny" , #  "openai/whisper-large-v3",
                torch_dtype=torch.float16,
                model_kwargs={"use_flash_attention_2": True},
                device="cuda:0")

outputs = pipe("d:/Downloads/sam_altman_lex_podcast_367.flac",
               chunk_length_s=30,
               batch_size=24,
               return_timestamps=True)

print(outputs["text"])            
'''