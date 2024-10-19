# Build Speech data files

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
import unicodedata
import re
import os
import pickle

df = pd.read_csv('data/audio_features.csv')
df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
print(df.shape)
display(df.head())

# change 7 to 2
df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
df.head()

df.to_csv('data/no_sample_df.csv')

# oversample fear
fear_df = df[df['label'] == 3]
for i in range(30):
    df = df.append(fear_df)

sur_df = df[df['label'] == 4]
for i in range(10):
    df = df.append(sur_df)

df.to_csv('data/modified_df.csv')

emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'neu': 3, }

scalar = MinMaxScaler()
df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])
df.head()

x_train, x_test = train_test_split(df, test_size=0.20)

x_train.to_csv('data/s2e/audio_train.csv', index=False)
x_test.to_csv('data/s2e/audio_test.csv', index=False)

print(x_train.shape, x_test.shape)


# Define preprocessing functions for text

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Build Text data files

useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

file2transcriptions = {}

for sess in range(1, 6):
    transcripts_path = 'data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
    transcript_files = os.listdir(transcripts_path)
    for f in transcript_files:
        with open('{}{}'.format(transcripts_path, f), 'r') as f:
            all_lines = f.readlines()

        for l in all_lines:
            audio_code = useful_regex.match(l).group()
            transcription = l.split(':')[-1].strip()
            # assuming that all the keys would be unique and hence no `try`
            file2transcriptions[audio_code] = transcription
# save dict
with open('data/t2e/audiocode2text.pkl', 'wb') as file:
    pickle.dump(file2transcriptions, file)
len(file2transcriptions)

audiocode2text = pickle.load(open('data/t2e/audiocode2text.pkl', 'rb'))

# Prepare text data
text_train = pd.DataFrame()
text_train['wav_file'] = x_train['wav_file']
text_train['label'] = x_train['label']
text_train['transcription'] = [normalizeString(audiocode2text[code]) for code in x_train['wav_file']]

text_test = pd.DataFrame()
text_test['wav_file'] = x_test['wav_file']
text_test['label'] = x_test['label']
text_test['transcription'] = [normalizeString(audiocode2text[code]) for code in x_test['wav_file']]

text_train.to_csv('data/t2e/text_train.csv', index=False)
text_test.to_csv('data/t2e/text_test.csv', index=False)

print(text_train.shape, text_test.shape)
