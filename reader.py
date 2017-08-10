import operator
import os


pad_token = '<pad>'
unk_token = '<unk>'


class FreqDist:
    def __init__(self, word_list):
        self.freq_list = {}
        for s in word_list:
            if s in self.freq_list:
                self.freq_list[s] += 1
            else:
                self.freq_list[s] = 1

    def most_common(self, num):
        sort_list = sorted(self.freq_list.items(), key=operator.itemgetter(1), reverse=True)
        return sort_list[:num]


def read_file(file_name):
    sent_list = []
    with open(file_name, 'r') as f:
        while True:
            s = f.readline()
            if not s: break
            sent_list.append(s.replace('\n', '').split(' '))
    return sent_list


def read_directory(dir_path):
    sent_list = []
    os.chdir(dir_path)
    files = os.listdir('.')
    for f in files:
        sent_list += read_file(f)
    return sent_list


def match_word_idx(sent_list, vocab_size):
    word_list = []
    for s in sent_list:
        word_list += s.split()
    
    freq_dist = FreqDist(word_list)
    freq_list = freq_dist.most_common(vocab_size)

    word2idx = {w: i+2 for i, (w, f) in enumerate(freq_list)}
    word2idx[pad_token] = 0
    word2idx[unk_token] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    return (word2idx, idx2word)
