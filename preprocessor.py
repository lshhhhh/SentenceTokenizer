import codecs
from konlpy.tag import Twitter
tagger = Twitter()
pad_token = '<pad>'
unk_token = '<unk>'


def tokenize(sent):
    return ['/'.join(t) for t in tagger.pos(sent, norm=True, stem=False)]


def match_token_idx(vocab):
    word2idx = {w: i+2 for i, (w, f) in enumerate(vocab)}
    word2idx[pad_token] = 0
    word2idx[unk_token] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    return (word2idx, idx2word)


def make_parallel_data(sent_list, word2idx):
    x_data = []
    y_data = []
    for s in sent_list:
        tmp = []
        for w in s:
            if not w in word2idx:
                w = unk_token
            tmp.append(str(word2idx[w]))
        x_data.append(tmp)
        tmp = []
        for i in range(len(s)-1):
            tmp.append('0')
        tmp.append('1')
        y_data.append(tmp)
    return (x_data, y_data)


def merge_sentences(x_data, y_data, num_token):
    x_merge = []
    y_merge = []
    tmp_x = []
    tmp_y = []
    for i, s in enumerate(x_data):
        if len(s) + len(tmp_x) <= num_token:
            tmp_x = tmp_x + s
            tmp_y = tmp_y + y_data[i]
        else:
            if len(tmp_x) <= num_token:
                x_merge.append(tmp_x)
                y_merge.append(tmp_y)
            tmp_x = s
            tmp_y = y_data[i]
    if len(tmp_x) <= num_token:
        x_merge.append(tmp_x)
        y_merge.append(tmp_y)

    return (x_merge, y_merge)
     
    
if __name__ == '__main__':
    '''
    Test
    '''
    in_file = codecs.open('./data/kr/kr.txt', 'r', 'utf-8')
    data = in_file.readlines()
    in_file.close()
    
    x_data, y_data = make_parallel_data(data)
    x_merge, y_merge = merge_sentences(x_data, y_data, 100)
    print(x_data[:2])
    print(x_merge[:2])
