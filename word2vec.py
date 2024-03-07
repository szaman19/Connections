import numpy as np
import gensim.downloader as api

wv = api.load("word2vec-google-news-300")


def word2vec(word, word_to_index, V):
    one_hot = np.zeros(V)
    index = word_to_index[word]
    one_hot[index] = 1
    return one_hot
