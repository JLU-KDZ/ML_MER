import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    'session_id': [1, 2, 3],
    'text': ['I am very happy today', 'I am so sad', 'I am extremely angry'],
    'emotion': ['Happy', 'Sad', 'Angry']
}
df = pd.DataFrame(data)

# 初始化TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer()

# 使用iemocap数据集的文本列来拟合TF-IDF模型并转换数据
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# 将TF-IDF矩阵转换为DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# 将原始DataFrame与TF-IDF特征DataFrame合并
df = pd.concat([df, tfidf_df], axis=1)

print(df)

