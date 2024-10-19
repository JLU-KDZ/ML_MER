import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# 加载预训练的wav2vec模型和处理器
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("wav2vec2-base-960h")

# 读取音频文件
audio_input, sample_rate = sf.read("data/IEMOCAP_full_release/Session1"
                                   "/dialog/wav/Ses01F_impro01.wav")

# 预处理音频数据
inputs = processor(audio_input, sampling_rate=sample_rate,
                   return_tensors="pt", padding=True)

# 提取特征
with torch.no_grad():
    features = model(**inputs).last_hidden_state

