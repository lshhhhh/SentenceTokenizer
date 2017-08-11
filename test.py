import numpy as np
import codecs
import stk
import preprocessor
import helpers
import tensorflow as tf


if __name__ == '__main__':
    #data_file = codecs.open('./data/kr/donga.kr.id.16k.bpe', 'r', 'utf-8')
    data_file = codecs.open('./data/simple/420.bpe', 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()

    epochs = 1000
    batch_size = 8
    seq_size = 70
    hidden_size = 64
    embedding_size = 64
    learning_rate = 0.1
    dic_size = 16000
    
    x_data, y_data = preprocessor.make_parallel_data(data)
    x_merge, y_merge = preprocessor.merge_sentences(x_data, y_data, seq_size)
    num_data = len(x_merge)

    print('Num of data: {}, {}'.format(np.array(x_data).shape, np.array(y_data).shape))
    print('Num of merged data: {}, {}'.format(np.array(x_merge).shape, np.array(y_merge).shape))
    
    train_file = codecs.open('./data/kr/train.kr', 'w', 'utf-8')
    test_file = codecs.open('./data/kr/test.kr', 'w', 'utf-8')
    num_test = int(num_data / 5)
    num_train = num_data - num_test
    for i in range(num_train):
        train_file.write(' '.join(x_merge[i])+'\n')
        train_file.write(' '.join(y_merge[i])+'\n')
    for i in range(num_test):
        test_file.write(' '.join(x_merge[i])+'\n')
        test_file.write(' '.join(y_merge[i])+'\n')
    train_file.close()
    test_file.close()

    stk = stk.SentenceTokenizer(
        batch_size=batch_size, seq_size=seq_size, dic_size=dic_size,
        hidden_size=hidden_size, embedding_size=embedding_size, learning_rate=learning_rate)

    train_files = ['./data/kr/train.kr']
    stk.train(train_files, epochs, num_train)

    test_files = ['./data/kr/test.kr']
    acc = stk.test(test_files, num_test)
    print('ACCURACY: ', acc)
    
