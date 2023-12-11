

import whisperx
import gc 
import torch
import time
import json 

HF_ACCESS_TOKEN= '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'
#from pyannote.audio import Pipeline
#Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=HF_ACCESS_TOKEN)

#Windows
device = "cuda" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model="large-v3"   #"small.en"  #"large-v3"
language="zh"
model = whisperx.load_model(model, device, compute_type=compute_type)

audio_filelist=[#"D:\Record/Family/record_20231126_1017.m4a",
                 #"D:\Record/Family/record_20231126_1038.m4a",
                  #"D:\Record/Family/record_20231126_1218.m4a",
                   #"D:\Record/Family/record_20231126_1354.m4a",
                    #"D:\Record/Family/record_20231126_1654.m4a",
                     #"D:\Record/Family/record_20231126_2230.m4a",
                      #"D:\Record/Family/record_20231127_0646.m4a",
                      "D:\Record/MyFamily/record_20231209_2016.m4a",
                       "D:\Record/Family/record_20231128_0655.m4a",
                        "D:\Record/Family/record_20231129_0646.m4a",
                         "D:\Record/Family/record_20231130_2020.m4a",
                          "D:\Record/Family/record_20231201_0647.m4a",
                           "D:\Record/Family/record_20231202_0837.m4a",
                            "D:\Record/Family/record_20231203_0703.m4a",
                                 "D:\Record/Family/record_20231209_0655.m4a",
                                 "D:\Record/Family/record_20231209_2016.m4a"
                                ]


#audio_file = "D:\Record/新录音 13.m4a"

#Mac M1
'''
device = "cpu"  # cuda or cpu
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # "float16" change to "int8" if low on GPU mem (may reduce accuracy)
audio_file = "/Users/henryking/Desktop/WhisperOutput/Unicom ANZ Meetings/UDTT meeting 2023-06-01/BTT0601.m4a"
'''
for audio_file in audio_filelist :
    print("\n\n####### audio_file ",audio_file)
    outputfile1=audio_file.split('.m4a')[0]+"_rawsegmt.json"
    outputfile2=audio_file.split('.m4a')[0]+ "_alignsegmt.json"
    outputfile3=audio_file.split('.m4a')[0]+ "_speakersegmt.json"

    # 1. Transcribe with original whisper (batched)

    model = whisperx.load_model(model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    
    # 记录程序开始时间
    start_time = time.time()

    result = model.transcribe(audio, batch_size=batch_size, language=language)
    #print(result["segments"]) # before alignment


    with open(outputfile1, 'w',encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)

    # delete model if low on GPU resources
    gc.collect(); 
    torch.cuda.empty_cache(); 
    del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    with open(outputfile2, 'w',encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)
    #print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    import gc; 
    gc.collect(); 
    torch.cuda.empty_cache(); 
    del model_a

    # 3. Assign speaker labels
    print("\n######## Assign speaker labels DiarizationPipeline ")
    #diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_ACCESS_TOKEN, device=device)
    diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization@2.1",use_auth_token=HF_ACCESS_TOKEN, device=device)


    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    #diarize_model(audio, min_speakers=2, max_speakers=5)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    #print(diarize_segments)
    #print(result["segments"]) # segments are now assigned speaker IDs

    gc.collect(); 
    torch.cuda.empty_cache(); 
    del diarize_model

    # 记录程序结束时间
    end_time = time.time()
    # 计算运行时间
    run_time = end_time - start_time
    # 打印运行时间
    print("程序运行时间：%.2f 秒" % run_time)


    with open(outputfile3, 'w',encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)


