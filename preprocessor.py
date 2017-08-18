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
    in_file = codecs.open('./data/kr/kr.rd', 'r', 'utf-8')
    #in_file = codecs.open('./data/simple/simple.txt.rd', 'r', 'utf-8')
    data = in_file.readlines()
    in_file.close()

    out_file = codecs.open('./data/kr/kr.rd.tk', 'w', 'utf-8')
    #out_file = codecs.open('./data/simple/simple.tk.txt', 'w', 'utf-8')
    for d in data:
        out_file.write(' '.join(tokenize(d))+'\n')
    out_file.close()
    
