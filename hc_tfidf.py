from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from sklearn.cluster import KMeans
import pandas as pd

# import tensorflow as tf
# import pylab as pl

# from tensorflow.contrib.factorization.python.ops import clustering_ops
# from mpl_toolkits.mplot3d import Axes3D

max_features = 1000
num_clusters = 8

# Define tokenizer
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

def create_tfidf(texts):
    # Create TF-IDF of texts
    tfidf = TfidfVectorizer(tokenizer=tokenizer, max_features=max_features)
    sparse_tfidf_texts = tfidf.fit_transform(texts) # len: 3954, 3954x1000
    # print(sparse_tfidf_texts.shape[1]) 
    # print(sparse_tfidf_texts.shape[0]) --> 3954

    # get similarity matrix
    similarity_matrix = cosine_similarity(sparse_tfidf_texts) # (3954, 3954)
    return (sparse_tfidf_texts, similarity_matrix)


def main():
    file_name = "2_contents.txt"
    corpus = open(file_name, 'r').read().split(',')
    tfidf_matrix, similarity_matrix = create_tfidf(corpus)
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    csv_input = pd.read_csv('data.csv')
    csv_input['category'] = cluster_list
    csv_input.to_csv('new_data.csv', index=False)

if __name__ == "__main__":
    # tf.app.run()
    main()