import numpy as np
import codecs
import datetime
import nltk
from gensim.models.word2vec import Word2Vec
from stk import *
from configs import *
from preprocessor import *


flags = tf.flags
flags.DEFINE_string(
    'model', 
    'small', 'A type of model. Possible options are: large, test.')
flags.DEFINE_string(
    'data_path', 
    None, 'The path to correctly-formatted data.')
flags.DEFINE_string(
    'test_sent',
    None,'The sentence to test')
flags.DEFINE_string(
    'test_data_path', 
    './data/result/test.kr', 'The path to correctly-formatted test data.')
flags.DEFINE_string(
    'lan',
    'ko', 'Language.')
FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == 'small':
        return SmallConfig()
    elif FLAGS.model == 'test':
        return TestConfig()
    else:
        return ValueError('Invalid model: ', FLAGS.model)


def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


if __name__ == '__main__':
    if not FLAGS.data_path:
        raise ValueError('Must set --data_path.')
    if not FLAGS.test_sent and not FLAGS.test_data_path:
        raise ValueError('Must set --test_sent or --test_data_path.')
    
    config = get_config()
    
    print('Read data file...')
    data_file = codecs.open(FLAGS.data_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()
    
    print('Build vocab...')
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
    
    if FLAGS.test_sent:
        sent = FLAGS.test_sent
        sent = tokenize(sent)
        pad = config.seq_size - (len(sent) % config.seq_size)
        for i in range(pad):
            sent = sent + [' ']
        test_sent = []
        for w in sent:
            if not w in word2idx:
                w = unk_token
            test_sent.append(word2idx[w])
        print('Sentences Test.')
        result = stk.test_sent(test_sent)
        print('Result: ', result)

    elif FLAGS.test_data_path:
        test_files = [FLAGS.test_data_path]
        num_test = int(file_len(FLAGS.test_data_path) / 2)
        print(num_test)
        print('File Test.')
        acc = stk.test_file(test_files, num_test)
        print('Accuracy: ', acc)

