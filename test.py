import numpy as np
import stk
import reader
import helpers


def is_special_mark(c):
    if c == '.' or c == ',' or c == '?' or c == '!' or c == '。':
        return True
    else:
        return False


def make_data(word2idx, data):
    sent_list = []
    for s in data:
        sent_list.append([w if w in word2idx else reader.unk_token for w in s])
    
    x_data = []
    x_mark = []
    for s in sent_list:
        if is_special_mark(s[-1]):
            x_mark.append(s[-1])
            s = s[:-1]
        else:
            x_mark.append(0)
        x_data.append([word2idx[w] for w in s])
    
    y_data = []
    for s in x_data:
        tmp = []
        for i in range(0, len(s)-1):
            tmp.append(0)
        tmp.append(1)
        y_data.append(tmp)
            
    x_data, x_size = helpers.batch(x_data)
    y_data, _ = helpers.batch(y_data)
    return (x_data, y_data, x_mark, x_size)


if __name__ == '__main__':
    '''
    Make train data.
    '''
    data = []
    data += reader.read_directory('/tmp/data1/users/dana/data/donga/enkr/kr-parallel_articles/kr_123.5')
    #data += reader.read_directory('/tmp/data1/users/dana/data/donga/enkr/kr-parallel_articles/kr_456.6')
    #data += reader.read_directory('/tmp/data1/users/dana/data/donga/enkr/kr-parallel_articles/kr_7890.7')
    print('Sentence number: ', len(data))
    
    word2idx, idx2word = reader.match_word_idx(data, 8000)
    x_train, y_train, x_mark, x_size = make_data(word2idx, data)
    
    batch_size = 16
    seq_size = x_size
    dic_size = len(word2idx)
    hidden_size = 256
    embedding_size = 64
    learning_rate = 0.1
    
    stk = stk.SentenceTokenizer(
        batch_size=batch_size, seq_size=seq_size, dic_size=dic_size,
        hidden_size=hidden_size, embedding_size=embedding_size, learning_rate=learning_rate)

    '''
    Train.
    '''
    stk.train(x_train, y_train)


    '''
    Test and evaluate accuracy.
    '''
    test_data = '▁기분 이 좋 다. ▁기분 이 좋 아 ▁난 무엇 을 ▁하 는 거지 ?'
    x_test, _, _, _ = make_data(word2idx, test_data)
    print(x_test)
    y_test = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    print(y_test)
    probs, result = stk.test(x_test)
    print(result)

    expect = y_test
    total_acc = 0.
    for i, e in enumerate(expect):
        if np.all(expect[i] == result[i]):
            total_acc += 1
    total_acc /= len(expect)
    total_acc *= 100
    print('ACCURACY: {}'.format(total_acc))

