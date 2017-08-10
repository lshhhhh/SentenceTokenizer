import sys
import codecs
import re
import random


def is_special_mark(c):
    if c == '.' or c == ',' or c == '?' or c == '!' or c == '。':
        return True
    else:
        return False


def make_parallel_data(sent_list):
    x_data = []
    x_mark = []
    for s in sent_list:
        s = re.sub('\n', '', s)
        if is_special_mark(s[-1]):
            if random.choice([True, False]):
                x_mark.append((s[-1], len(s)-1))
                s = s[:-1]
            else:
                x_mark.append((0, None))
        else:
            x_mark.append((0, None))
        x_data.append(s)
    
    y_data = []
    for s in x_data:
        s = s.split()
        tmp = []
        for i in range(0, len(s)-1):
            tmp.append('0')
        tmp.append('1')
        y_data.append(tmp)

    return (x_data, y_data, x_mark)


def merge_data(x_data, y_data, num_token):
    x_merge = []
    y_merge = []
    tmp_x = []
    tmp_y = []
    for i, s in enumerate(x_data):
        s = s.split()
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
    lan = sys.argv[1]
    #in_file = codecs.open('./data/{}/all.{}.bpe'.format(lan, lan), 'r', 'utf-8')
    in_file = codecs.open('./data/9.kr', 'r', 'utf-8')
    data = in_file.readlines()
    in_file.close()
    
    x_data, y_data, _ = make_parallel_data(data)
    x_merge, y_merge = merge_data(x_data, y_data, 100)
    print(x_data)
    print(x_merge)
