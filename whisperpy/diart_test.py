
#https://github.com/juanmc2005/diart

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
import huggingface_hub
import os
os.environ["HUGGINGFACE_TOKEN"] ='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'

huggingface_hub.login()
pipeline = SpeakerDiarization()
mic = MicrophoneAudioSource()
inference = StreamingInference(pipeline, mic, do_plot=True)
inference.attach_observers(RTTMWriter(mic.uri, "/Users/henryking/Desktop/diartoutputfile.rttm"))
prediction = inference()