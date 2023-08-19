import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('sentence-transformers/LaBSE')

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它

    return para.split("\n")
def cut_sent_en(para):
  para = re.sub(
      r'(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<=\.|\?|\!)\"*\s*\s*\W*(?<=[.!?])\s*(?=[A-Z])',
      "\n", para)
  para = re.sub('([.]\s*["\'])', r'\1\n', para)
  # para = re.sub('([.!\?]\s*["\'])([^.!\?])', r'\1\n\2', para)
  para = para.rstrip()
  return para.split("\n")


data = pd.read_excel("demo.xlsx")
df_s = pd.DataFrame(data)
similar_sentences = pd.DataFrame(columns=["Sentence1", "Sentence2", "Similarity"])  # 在循环外部初始化

for paragraph in range(len(df_s)):
    chinese_sentences_list = cut_sent(df_s.loc[paragraph][1])
    print(chinese_sentences_list)
    english_sentences_list = cut_sent_en(df_s.loc[paragraph][0])

    chinese_len = len(chinese_sentences_list)
    english_len = len(english_sentences_list)
    embeddings1 = model.encode(chinese_sentences_list)
    embeddings2 = model.encode(english_sentences_list)
    match=[]
    for zh in range(chinese_len):
        for en in range(english_len):
            if en in match:
                continue
            dot_product = np.dot(embeddings1[zh], embeddings2[en])
            norm_vector1 = np.linalg.norm(embeddings1[zh])
            norm_vector2 = np.linalg.norm(embeddings2[en])
            similarity = dot_product / (norm_vector1 * norm_vector2)
            if similarity > 0.6:
                match.append(en)
                similar_sentences = similar_sentences.append(
                    {"Sentence1": chinese_sentences_list[zh], "Sentence2": english_sentences_list[en],
                     "Similarity": similarity}, ignore_index=True)  # 在循环内部追加
                break
# 将相似句子保存到CSV文件
similar_sentences.to_csv("similar_sentences.csv", encoding='utf-8-sig', index=False)