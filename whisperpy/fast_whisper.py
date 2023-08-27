from faster_whisper import WhisperModel
import time

# 记录程序开始时间
start_time = time.time()

#model_size = "large-v2"   # 3G
#model_size  = "medium"     # 1.5G  working after re-install faster-whisper package. 

model_size  = "large-v2"

# Run on CPU with INT8 
model = WhisperModel(model_size, device="cpu", compute_type="int8")
 

# or run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

#segments, info = model.transcribe("/Users/henryking/Desktop/WhisperOutput/Home/Home 0622_2.m4a", language="zh" , beam_size=5)

full_fileName="/Users/henryking/Downloads/ChatUDTTDocMeeting0808.m4a"
output_file="/Users/henryking/Downloads/ChatUDTTDocMeeting0808.txt"
output_mdfile="/Users/henryking/Downloads/ChatUDTTDocMeeting0808.md"
# vad_filter=True   Silero VAD model to filter out parts of the audio without speech, the default is more than 2s
# 缺省的 beam_size=5 对于 small 和 medium 在M1上最优值，调到8反而带来时间变长，质量下降 , 对large似乎beam=8最佳效果，但时间多50%
segments, info = model.transcribe(full_fileName , beam_size=5 ,vad_filter=True) 

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

contents=[]
with open(output_file, "w") as file:
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # 将 segment 的内容写入文件
        file.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
        contents.append(segment.text)
# 关闭文件
file.close()

# 记录程序结束时间
end_time = time.time()
# 计算运行时间
run_time = end_time - start_time
# 打印运行时间
print("程序运行时间：%.2f 秒" % run_time)

 
        
with open(output_mdfile, "w") as file:
    # 遍历每个 segment
    for record in contents:
        # 将 segment 的内容写入文件
        file.write("%s\n" % ( record))

# 关闭文件
file.close()



 