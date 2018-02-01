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

max_features = 1500
num_clusters = 5

# Define tokenizer
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

def create_tfidf(texts):
    # Create TF-IDF of texts
    tfidf = TfidfVectorizer(tokenizer=tokenizer, max_features=max_features)
    sparse_tfidf_texts = tfidf.fit_transform(texts) # len: 3954, 3954x1500

    # get similarity matrix
    similarity_matrix = cosine_similarity(sparse_tfidf_texts) # (3954, 3954)
    return (sparse_tfidf_texts, similarity_matrix)


def main():
    file_name = "0_contents.txt"
    corpus = open(file_name, 'r').read().split(',')
    #   vectorize the corpus
    tfidf_matrix, similarity_matrix = create_tfidf(corpus)
    #   apply k-means clustering method
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    #   save the assigned cluster
    csv_input = pd.read_csv('data.csv')
    csv_input['category'] = cluster_list
    csv_input.to_csv('0_data.csv', index=False)
    #   save the cosine similarity matrix for later use
    np.save('simlarity', np.array(similarity_matrix))

if __name__ == "__main__":
    # tf.app.run()
    main()
