import codecs
import preprocessor as pp
from gensim.models.word2vec import Word2Vec


if __name__ == '__main__':
    file_path = './data/kr/kr.tk'
    data_file = codecs.open(file_path, 'r', 'utf-8')
    data = data_file.readlines()
    data_file.close()

    sent_list = [d.split() for d in data] 
    model = Word2Vec(sent_list, min_count=5, size=64)
    model.init_sims(replace=True)
    model.save('./tmp/word2vec.model')
    #model.train(more_sentences)
