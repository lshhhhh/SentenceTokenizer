import codecs
import re
import random
import sys


def is_special_mark(c):
    if c == '.' or c == ',' or c == '?' or c == '!' or c == '。':
        return True
    else:
        return False


def random_drop_special_char(file_path):
    in_file = codecs.open(file_path, 'r', 'utf-8')
    out_file = codecs.open(file_path+'.rd', 'w', 'utf-8')
    data = in_file.readlines()
    for s in data:
        s = re.sub('\n', '', s)
        if (len(s) > 0):
            if is_special_mark(s[-1]):
                if random.choice([True, False]):
                    s = s[:-1]
                    print("CHANGE: ", s)
            out_file.write(s+'\n')
    in_file.close()
    out_file.close()


if __name__ == '__main__':
    file_path = sys.argv[1]
    random_drop_special_char(file_path)
