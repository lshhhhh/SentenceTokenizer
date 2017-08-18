import tensorflow as tf
import numpy as np
import codecs
import datetime
import nltk

from gensim.models.word2vec import Word2Vec
from stk import *
from configs import *
from preprocessor import *


flags = tf.flags
flags.DEFINE_string('model', 'test', 'A type of model. Possible options are: large, test.')
flags.DEFINE_string('data_path', './data/simple/simple.rd.tk', 'The path to correctly-formatted data.')
flags.DEFINE_boolean('tensorboard', False, 'Whether to write data to a TensorBoard summary.')
flags.DEFINE_integer('vocab_size', 8000, 'Vocaburaly size.')

FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == 'large':
        return LargeConfig()
    elif FLAGS.model == 'test':
        return TestConfig()
    else:
        return ValueError('Invalid model: ', FLAGS.model)


if __name__ == '__main__':
    if not FLAGS.data_path:
        raise ValueError('Must set --data_path.')
    
    config = get_config()
    if FLAGS.vocab_size:
        config.vocab_size = FLAGS.vocab_size
    
    train_file_path = './data/result/train.kr'
    test_file_path = './data/result/test.kr'
    
    result_file = open('./data/result.kr', 'w')
    result_file.write(FLAGS.data_path+'\nStart: '+str(datetime.datetime.now())+'\n')
   
    data_file = codecs.open(FLAGS.data_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()
    
    sent_list = [d.split() for d in data]
    tokens = [t for sent in sent_list for t in sent]
    text = nltk.Text(tokens, name='Donga articles')
    vocab = text.vocab().most_common(config.vocab_size-1)
    word2idx, idx2word = match_token_idx(vocab)
    
    model = Word2Vec.load('./tmp/word2vec.model')
    embeddings = np.zeros((config.vocab_size, config.embedding_size))
    for (w, i) in word2idx.items():
        try:
            embeddings[i] = model.wv[w]
        except:
            pass
    
    x_data, y_data = make_parallel_data(sent_list, word2idx)
    x_merge, y_merge = merge_sentences(x_data, y_data, config.seq_size)
    result_file.write('Num of data: {}, {}\n'.format(np.array(x_data).shape, np.array(y_data).shape)+
                      'Num of merged data: {}, {}\n'.format(np.array(x_merge).shape, np.array(y_merge).shape))
    
    train_file = codecs.open(train_file_path, 'w', 'utf-8')
    test_file = codecs.open(test_file_path, 'w', 'utf-8')
    num_data = len(x_merge)
    num_test = int(num_data / 5)
    num_train = num_data - num_test
    for i in range(num_test):
        test_file.write(' '.join(x_merge[i])+'\n'+' '.join(y_merge[i])+'\n')
    for i in range(num_test, num_data):
        train_file.write(' '.join(x_merge[i])+'\n'+' '.join(y_merge[i])+'\n')
    train_file.close()
    test_file.close()

    stk = SentenceTokenizer(config=config, embeddings=embeddings)

    train_files = [train_file_path]
    stk.train(train_files, config.epochs, num_train)

    test_files = [test_file_path]
    acc = stk.test(test_files, num_test)
    
    print('Accuracy: ', acc)
    result_file.write('End: '+str(datetime.datetime.now())+'\n'+'Accuracy: '+str(acc))
    result_file.close()
    
