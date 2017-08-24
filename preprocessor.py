#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import codecs
import nltk
import numpy as np
from configs import *
from konlpy.tag import Twitter

tagger = Twitter()
pad_token = '<pad>'
unk_token = '<unk>'


def tokenize(sent):
    return ['/'.join(t) for t in tagger.pos(sent, norm=True, stem=False)]


def build_dict(sent_list, config):
    tokens = [t for sent in sent_list for t in sent]
    text = nltk.Text(tokens, name='text')
    vocab = text.vocab().most_common(config.vocab_size-2)
    word2idx = {w: i+2 for i, (w, f) in enumerate(vocab)}
    word2idx[pad_token] = 0
    word2idx[unk_token] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    return (word2idx, idx2word)


def make_parallel_data(sent_list, word2idx):
    x_data = []
    y_data = []
    for s in sent_list:
        if len(s) > 0:
            tmp_x = []
            tmp_y = []
            for w in s:
                if not w in word2idx:
                    w = unk_token
                tmp_x.append(str(word2idx[w]))
            for i in range(len(s)):
                if i == len(s) - 1:
                    tmp_y.append('1')
                else:
                    tmp_y.append('0')
            if len(tmp_x) == len(tmp_y):
                x_data.append(tmp_x)
                y_data.append(tmp_y)
            else:
                print('x, y not match!')
                print(s)
                break
    return (x_data, y_data)


def divide_sentences(x_data, y_data, num_token):
    x_flatten = [word for sent in x_data for word in sent]
    y_flatten = [word for sent in y_data for word in sent]
    x_divide = []
    y_divide = []
    for i in range(len(x_flatten)/num_token):
        x_divide.append(x_flatten[i*num_token:(i+1)*num_token])
        y_divide.append(y_flatten[i*num_token:(i+1)*num_token])
    return (x_divide, y_divide)
     
