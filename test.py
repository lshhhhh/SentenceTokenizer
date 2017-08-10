import numpy as np
import codecs
import stk
import preprocessor
import reader
import helpers
import tensorflow as tf


if __name__ == '__main__':
    data_file = codecs.open('./data/9.kr', 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()

    epochs = 1000
    batch_size = 10
    seq_size = 70
    hidden_size = 64
    embedding_size = 64
    learning_rate = 0.1
    
    word2idx, idx2word = reader.match_word_idx(data, 10000)
    dic_size = len(word2idx)
    
    x_data, y_data, _ = preprocessor.make_parallel_data(data)
    x_merge, y_merge = preprocessor.merge_data(x_data, y_data, seq_size)
    num_data = len(x_merge)

    #print('Num of data: {}, {}'.format(np.array(x_data).shape, np.array(y_data).shape))
    #print('Num of merged data: {}, {}'.format(np.array(x_merge).shape, np.array(y_merge).shape))
    
    train_file = codecs.open('./data/kr/train.kr', 'w', 'utf-8')
    test_file = codecs.open('./data/kr/test.kr', 'w', 'utf-8')
    num_test = int(num_data / 5)
    num_train = num_data - num_test
    for i in range(num_train):
        x_idx = []
        for w in x_merge[i]:
            x_idx.append(str(word2idx[w]))
        train_file.write(' '.join(x_idx)+'\n')
        train_file.write(' '.join(y_merge[i])+'\n')
    for i in range(num_test):
        x_idx = []
        for w in x_merge[i]:
            x_idx.append(str(word2idx[w]))
        test_file.write(' '.join(x_idx)+'\n')
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
    
