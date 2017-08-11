import sys
import os
import codecs


if __name__ == '__main__':
    source_dir = sys.argv[1]
    lan = sys.argv[2]
    
    out_file = codecs.open('./{}/all.{}'.format(lan, lan), 'a', 'utf-8')
    os.chdir(source_dir)
    files = os.listdir('.')
    for f in files:
        in_file = codecs.open('{}/{}'.format(source_dir, f), 'r', 'utf-8')
        data = in_file.readlines()
        for s in data:
            out_file.write(s)
        in_file.close()
    out_file.close()
