import subprocess

program = "/Users/henryking/whisper.cpp/main"
model = "/Users/henryking/whisper.cpp/models/ggml-base.en-q5_0.bin"
audio_file = "/Users/henryking/whisper.cpp/BTT0601.WAV"

print("start ..calling :  /Users/henryking/whisper.cpp/main -m  /Users/henryking/whisper.cpp/models/ggml-base.en-q5_0.bin  /Users/henryking/whisper.cpp/BTT0601.WAV")

result = subprocess.run([program, "-m", model, audio_file], capture_output=True, text=True)

subprocess.run([program, "-m", model, audio_file], capture_output=True, text=True)


if result.returncode == 0:
    print(result.stdout)
else:
    print(result.stderr)