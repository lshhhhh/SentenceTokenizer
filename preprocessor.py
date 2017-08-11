import sys
import codecs
import re
import random


def make_parallel_data(sent_list):
    x_data = []
    y_data = []
    for s in sent_list:
        s = s.split()
        x_data.append(s)
        tmp = []
        for i in range(0, len(s)-1):
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
        if len(s) + len(tmp_x) < num_token:
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
    in_file = codecs.open('./kr/donga.kr.id.16k.bpe', 'r', 'utf-8')
    data = in_file.readlines()
    in_file.close()
    
    x_data, y_data = make_parallel_data(data)
    x_merge, y_merge = merge_sentences(x_data, y_data, 100)
    print(x_data[:2])
    print(x_merge[:2])
