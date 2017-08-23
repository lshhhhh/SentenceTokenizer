import numpy as np
import codecs
import datetime
import nltk
from gensim.models.word2vec import Word2Vec
from stk import *
from configs import *
from preprocessor import *


def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


if __name__ == '__main__':
    config = LargeConfig()
    data_file_path = './data/kr/kr.rd.tk'
    test_file_path = './data/result/test.kr'
    
    print('Read file...')
    data_file = codecs.open(data_file_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()
    
    print('Build Vocab...')
    sent_list = [d.split() for d in data]
    word2idx, idx2word = build_dict(sent_list, config)
    
    print('Load word2vec...')
    model = Word2Vec.load('./tmp/word2vec.model')
    embeddings = np.zeros((config.vocab_size, config.embedding_size))
    for (w, i) in word2idx.items():
        try:
            embeddings[i] = model.wv[w]
        except:
            pass

    stk = SentenceTokenizer(config=config, embeddings=embeddings)

    test_files = [test_file_path]
    num_test = int(file_len(test_file_path) / 2)
    print(num_test)
    print('Test Start.')
    acc = stk.test(test_files, num_test)
    print('Accuracy: ', acc)

