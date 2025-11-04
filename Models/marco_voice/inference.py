import os
import sys

import torch
import torchaudio

from cosyvoice_rodis.cli.cosyvoice import CosyVoice
# from Models.marco_voice.cosyvoice_emosphere.cli.cosyvoice import CosyVoice as cosy_emosphere

from cosyvoice_rodis.utils.file_utils import load_wav
# sys.path.append("PathtoModels/marco_voice")
# Load pre-trained model
model = CosyVoice('your_model_path/models_path/CosyVoice-300M-rodis', load_jit=False, load_onnx=False, fp16=False)
emo = {"伤心": "Sad", "恐惧":"Fearful", "快乐": "Happy", "惊喜": "Surprise", "生气": "Angry", "戏谑":"Jolliest"} 
prompt_speech_16k = load_wav("your prompt audio path", 16000)
emo_type="快乐"
if emo_type in ["生气", "惊喜", "快乐"]:
    emotion_info = torch.load("../../assets/emotion_info.pt")["male005"][emo.get(emo_type)] 
elif emo_type in ["伤心"]:
    emotion_info = torch.load("../../assets/emotion_info.pt")["female005"][emo.get(emo_type)]
elif emo_type in ["恐惧"]:
    emotion_info = torch.load("../../assets/emotion_info.pt")["female003"][emo.get(emo_type)]
else:
    emotion_info = torch.load("../../assets/emotion_info.pt")["male005"][emo.get(emo_type)]
# Voice cloning with discrete emotion
for i, j in enumerate(model.synthesize(
    tts_text="今天的天气真不错，我们出去散步吧！",
    prompt_text="your reference audio content",
    prompt_speech_16k=prompt_speech_16k,
    key=emo_type,
    emotion_embedding=emotion_info
)):
  torchaudio.save('emotional_{}.wav'.format(emo_type), j['tts_speech'], 22050)
# tts_text, prompt_text, prompt_speech_16k, key, emotion_speakerminus
# Continuous emotion control
# model_emosphere = cosy_emosphere('your_model_path/models_path/CosyVoice-300M-rodis', load_jit=False, load_onnx=False, fp16=False)

# for i, j in enumerate(model_emosphere.synthesize(
#     text="今天的天气真不错，我们出去散步吧！",
#     prompt_text="",
#     reference_speech=prompt_speech_16k,
#     emotion_embedding=emotion_info,
#     low_level_emo_embedding=[0.1, 0.4, 0.5]
# )):
#   torchaudio.save('emosphere_{}.wav'.format(emo_type), j['tts_speech'], 22050)



### More Features
# Cross-lingual emotion transfer
# for i, j in enumerate(model.synthesize(
#     text="hello, i'm a speech synthesis model, how are you today? ",
#     prompt_text="",
#     reference_speech=prompt_speech_16k,
#     emo_type=emo_type,
#     emotion_embedding=emotion_info
# )):
#   torchaudio.save('emosphere_ross_lingual_{}.wav'.format(emo_type), j['tts_speech'], 22050)