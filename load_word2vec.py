from gensim.models import word2vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('/newvolume/text8')
model = word2vec.Word2Vec(sentences, size=200)
model.save('/newvolume/text8.model')
model.wv.save_word2vec_format('/newvolume/text.model.bin', binary=True)
