from sentence_transformers import SentenceTransformer, models
import numpy as np
from bitext_mining_utils import *
import gzip
import tqdm
from sklearn.decomposition import PCA
import torch
import csv
import pandas as pd
import re

def cut_sent_de(para):
    para = re.sub('([…,:;?!])', r'\1\n', para)
    para = re.sub(r'\.\s(?=[A-Z])', '\n', para)
    para = para.rstrip()
    return para.split("\n")
def cut_sent_en(para):
    para = re.sub('([,:;；?!])', r'\1\n', para)
    para = re.sub(r'\.\s(?=[A-Z])', '\n', para)
    para = para.rstrip()
    return para.split("\n")


chunk_size = 5000  # 每个文件的行数
with open("predict.de", encoding='utf-8') as fd:
    lines = fd.readlines()
total_lines = len(lines)
num_files = total_lines // chunk_size + 1  # 计算需要创建的文件数

for i in range(num_files):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size
    chunk_lines = lines[start_index:end_index]

    output_file = f"v16-de/v16-de-{i+1}.txt"  # 根据索引创建新的文件名
    with open(output_file, "w", encoding='utf-8') as fd:
        fd.writelines(chunk_lines)

with open("source.en", encoding='utf-8') as fe:
    lines = fe.readlines()

for i in range(num_files):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size
    chunk_lines = lines[start_index:end_index]
    output_file = f"v16-en/v16-en-{i+1}.txt"  # 根据索引创建新的文件名
    with open(output_file, "w", encoding='utf-8') as f:
        f.writelines(chunk_lines)
for num_txt in range(1,num_files+1):
    str1="v16-de/v16-de-"+str(num_txt)+".txt"
    str2="v16-en/v16-en-"+str(num_txt)+".txt"
    with open(str1, 'r',encoding='utf8') as f:
        data1 = [line.strip() for line in f.readlines()]
    df1 = pd.DataFrame(data1, columns=['a'],)

    with open(str2, 'r',encoding='utf8') as f:
        data2 = [line.strip() for line in f.readlines()]
    df2 = pd.DataFrame(data2, columns=['b'])
    dfx = pd.concat([df1, df2], axis=1)
    str3="v16-de/v16-de-s"+str(num_txt)+".txt"
    str4="v16-en/v16-en-s"+str(num_txt)+".txt"
    with open(str3, 'w',encoding='utf8') as file:
        with open(str4, 'w', encoding='utf8') as f:
            for i in range(len(dfx)):
                sent1 = str(dfx.loc[i][0])
                sent2 = str(dfx.loc[i][1])
                sent_zh = cut_sent_de(sent1)
                len_zh = len(sent_zh)
                sent_en = cut_sent_en(sent2)
                len_en = len(sent_en)
                if len_en==1 and len_zh==1:
                    continue
                list_z=[]
                list_e=[]
                for num_z in range(1, len_zh + 1):  # 控制字符串数量
                    for j in range(len_zh - num_z + 1):  # 控制字符串起始位置
                        list_z.append(''.join(sent_zh[j:j + num_z]))
                for num_e in range(1, len_en + 1):  # 控制字符串数量
                    for j in range(len_en - num_e + 1):  # 控制字符串起始位置
                        list_e.append(''.join(sent_en[j:j + num_e]))
                list_z=[item for item in list_z if len(item) >= 5 and len(item)<=200]
                list_e=[item for item in list_e if len(item) >= 11 and len(item)<=300]

                for item in list_z:
                    file.write("%s\n" % item)
                for item in list_e:
                    f.write("%s\n" % item)
model_name = 'LaBSE'
model = SentenceTransformer(model_name)
for un_num in range(1,num_files+1):
    #Input files. We interpret every line as sentence.
    str1 = "v16-de/v16-de-s"+str(un_num)+".txt"
    str2 = "v16-en/v16-en-s"+str(un_num)+".txt"
    source_file = str1
    target_file = str2

    min_sent_len = 10
    max_sent_len = 200
    knn_neighbors = 4
    min_threshold = 1.05
    use_ann_search = False
    ann_num_clusters = 32768
    ann_num_cluster_probe = 3
    use_pca = False
    pca_dimensions = 128
    print("Read source file")
    source_sentences = set()
    with file_open(source_file) as fIn:
        for line in tqdm.tqdm(fIn):
            line = line.strip()
            if len(line) >= 15 and len(line) <= max_sent_len:
                source_sentences.add(line)

    print("Read target file")
    target_sentences = set()
    with file_open(target_file) as fIn:
        for line in tqdm.tqdm(fIn):
            line = line.strip()
            if len(line) >= 15 and len(line) <= max_sent_len:
                target_sentences.add(line)

    print("Source Sentences:", len(source_sentences))
    print("Target Sentences:", len(target_sentences))

    ### Encode source sentences
    source_sentences = list(source_sentences)

    print("Encode source sentences")
    source_embeddings = model.encode(source_sentences, show_progress_bar=True, convert_to_numpy=True)
    target_sentences = list(target_sentences)
    print("Encode target sentences")
    target_embeddings = model.encode(target_sentences, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings
    x = source_embeddings
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    y = target_embeddings
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Perform kNN in both directions
    x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
    x2y_mean = x2y_sim.mean(axis=1)

    y2x_sim, y2x_ind = kNN(y, x, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
    y2x_mean = y2x_sim.mean(axis=1)

    # Compute forward and backward scores
    margin = lambda a, b: a / b
    fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
    bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

    indices = np.stack([np.concatenate([np.arange(x.shape[0]), bwd_best]), np.concatenate([fwd_best, np.arange(y.shape[0])])], axis=1)
    scores = np.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
    seen_src, seen_trg = set(), set()
    print("Write sentences to disc")
    sentences_written = 0
    str3 = 'result/v16-de-en'+str(un_num)+'.csv'
    with open(str3, 'wt', encoding='utf-8-sig', newline='') as fOut:
        csv_writer = csv.writer(fOut, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['a', 'b', 'c'])
        for i in np.argsort(-scores):
            src_ind, trg_ind = indices[i]
            src_ind = int(src_ind)
            trg_ind = int(trg_ind)
            if scores[i] < min_threshold:
                break
            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                csv_writer.writerow([scores[i], source_sentences[src_ind].replace(",", " "),
                                     target_sentences[trg_ind].replace(",", " ")])
                sentences_written += 1
    print("Done. {} sentences written".format(sentences_written))
file_paths = []
for i in range(1,num_files+1):
    file_paths.append(f"result/v16-de-en{i}.csv")
df_list = []  # 存储DataFrame的列表
for file_path in file_paths:
    df = pd.read_csv(file_path, encoding='utf8')
    df_list.append(df)

combined_df = pd.concat(df_list) 
sorted_df = combined_df.sort_values(by="a", ascending=False)
sorted_df=sorted_df.drop_duplicates()
sorted_df = sorted_df.head(600000)
# 将排序后的DataFrame写入新的CSV文件
sorted_df.to_csv('result/平行子句.csv',encoding='utf-8-sig', index=False)