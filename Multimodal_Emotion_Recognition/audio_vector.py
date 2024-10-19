# Build Audio Vectors

import librosa
import os
import soundfile as sf
import numpy as np
import matplotlib.style as ms
from tqdm import tqdm
import pickle
import pandas as pd
import math
import librosa.display
ms.use('seaborn-muted')


file_path = 'data/IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav'
y, sr = librosa.load(file_path, sr=44100)

labels_df = pd.read_csv('data/pre-processed/df_iemocap.csv')
iemocap_dir = 'data/IEMOCAP_full_release/'


audio_vectors = {}
for sess in [5]:  # using one session due to memory constraint, can replace [5] with range(1, 6)
    wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                audio_vectors[truncated_wav_file_name] = truncated_wav_vector
        except:
            print('An exception occured for {}'.format(orig_wav_file))
    with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
        pickle.dump(audio_vectors, f)