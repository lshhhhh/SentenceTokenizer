import numpy as np
import codecs
import stk
import preprocessor as pp
import datetime
import nltk
from gensim.models.word2vec import Word2Vec


if __name__ == '__main__':
    file_path = './data/kr/kr.txt'
    #file_path = './data/simple/simple.txt.rd'
    train_file_path = './data/train.kr.txt'
    test_file_path = './data/test.kr.txt'
    
    result_file = open('./data/result.kr.txt', 'w')
    result_file.write(file_path+'\n')
    result_file.write('Start: ')
    result_file.write(str(datetime.datetime.now())+'\n')
   
    epochs = 1
    batch_size = 16
    seq_size = 70
    hidden_size = 64
    embedding_size = 64
    learning_rate = 0.1
    vocab_size = 16000
    
    data_file = codecs.open(file_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()

    sent_list = [pp.tokenize(d) for d in data]
    tokens = [t for sent in sent_list for t in sent]
    text = nltk.Text(tokens, name='Donga articles')
    vocab = text.vocab().most_common(vocab_size-1)
    word2idx, idx2word = pp.match_token_idx(vocab)
   
    model = Word2Vec.load('./tmp/word2vec.model')
    embeddings = np.zeros((vocab_size, embedding_size))
    for (w, i) in word2idx.items():
        try:
            embeddings[i] = model.wv[w]
        except:
            pass

    x_data, y_data = pp.make_parallel_data(sent_list, word2idx)
    x_merge, y_merge = pp.merge_sentences(x_data, y_data, seq_size)
    num_data = len(x_merge)

    result_file.write('Num of data: {}, {}\n'.format(np.array(x_data).shape, np.array(y_data).shape))
    result_file.write('Num of merged data: {}, {}\n'.format(np.array(x_merge).shape, np.array(y_merge).shape))
    
    train_file = codecs.open(train_file_path, 'w', 'utf-8')
    test_file = codecs.open(test_file_path, 'w', 'utf-8')
    num_test = int(num_data / 5)
    num_train = num_data - num_test
    for i in range(num_test):
        test_file.write(' '.join(x_merge[i])+'\n')
        test_file.write(' '.join(y_merge[i])+'\n')
    for i in range(num_test, num_data):
        train_file.write(' '.join(x_merge[i])+'\n')
        train_file.write(' '.join(y_merge[i])+'\n')
    train_file.close()
    test_file.close()

    stk = stk.SentenceTokenizer(
        batch_size=batch_size, seq_size=seq_size, vocab_size=vocab_size,
        hidden_size=hidden_size, embedding_size=embedding_size, learning_rate=learning_rate,
        embeddings=embeddings)

    train_files = [train_file_path]
    stk.train(train_files, epochs, num_train)

    test_files = [test_file_path]
    acc = stk.test(test_files, num_test)
    
    result_file.write('End: ')
    result_file.write(str(datetime.datetime.now())+'\n')
    result_file.write('Accuracy: '+str(acc))
    result_file.close()
    print('Accuracy: ', acc)

