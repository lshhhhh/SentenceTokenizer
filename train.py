import tensorflow as tf
import numpy as np
import codecs
from gensim.models.word2vec import Word2Vec
from stk import *
from configs import *
from preprocessor import *


flags = tf.flags
flags.DEFINE_string('model', 
                    'test', 
                    'A type of model. Possible options are: large, test.')
flags.DEFINE_string('data_path', 
                    './data/simple/simple.rd.tk', 
                    'The path to correctly-formatted data.')
flags.DEFINE_boolean('tensorboard', 
                     False, 
                     'Whether to write data to a TensorBoard summary.')
flags.DEFINE_integer('vocab_size', 
                     8000, 
                     'Vocaburaly size.')
FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == 'small':
        return SmallConfig()
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
    
    data_file = codecs.open(FLAGS.data_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()
    
    sent_list = [d.split() for d in data]
    word2idx, idx2word = build_dict(sent_list, config)
    
    model = Word2Vec.load('./tmp/word2vec.model')
    embeddings = np.zeros((config.vocab_size, config.embedding_size))
    for (w, i) in word2idx.items():
        try:
            embeddings[i] = model.wv[w]
        except:
            pass
    
    x_data, y_data = make_parallel_data(sent_list, word2idx)
    x_divide, y_divide = divide_sentences(x_data, y_data, config.seq_size)
    
    train_file = codecs.open(train_file_path, 'w', 'utf-8')
    test_file = codecs.open(test_file_path, 'w', 'utf-8')
    num_data = len(x_divide)
    if num_data < 5000:
        num_test = int(num_data / 5)
    else:
        num_test = 1000
    num_train = num_data - num_test

    for i in range(num_test):
        test_file.write(' '.join(x_divide[i])+'\n'+' '.join(y_divide[i])+'\n')
    for i in range(num_test, num_data):
        train_file.write(' '.join(x_divide[i])+'\n'+' '.join(y_divide[i])+'\n')
    train_file.close()
    test_file.close()

    stk = SentenceTokenizer(config=config, embeddings=embeddings)

    train_files = [train_file_path]
    stk.train(train_files, config.epochs, num_train)
    

