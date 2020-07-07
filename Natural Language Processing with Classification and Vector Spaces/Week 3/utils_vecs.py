# This is used to tranlate english to french

import pandas as pd
from gensim.models import KeyedVectors
import nltk
import unicodedata
import string

# Loading in the French embeddings. 

fr_embeddings = KeyedVectors.load_word2vec_format('wiki.multi.fr.vec')
f = open('capitals.txt', 'r').read()
set_words = set(nltk.word_tokenize(f))

def load_translations():
    '''
    TBD
    
    '''
    dict_fr = pd.read_csv('en-fr.txt', delimiter = ' ')
    
    en_to_fr = {}
    fr_to_vec = {}
    for i in range(len(dict_fr)):
        en = dict_fr.loc[i][0]
        fr = dict_fr.loc[i][1]
        if type(en) != float:
            en = en.capitalize()
        if en in set_words and en not in set(en_to_fr.keys()):
            en_to_fr[en] = fr
            fr_to_vec[fr] = fr_embeddings[fr]
    # Add comments later 
    del fr_to_vec['syrienne']
    del fr_to_vec['iranienne']
    del fr_to_vec['malien']
    del fr_to_vec['arménienne']
    del fr_to_vec['chilien']
    del fr_to_vec['équateur']
    en_to_fr['Chile'] = 'chili'
    fr_to_vec['chili'] = fr_embeddings['chili']
    en_to_fr['Iran'] = 'iran'
    fr_to_vec['iran'] = fr_embeddings['iran']
    en_to_fr['Turkey'] = 'turquie'
    fr_to_vec['turquie'] = fr_embeddings['turquie']
    en_to_fr['Syria'] = 'syrie'
    fr_to_vec['syrie'] = fr_embeddings['syrie']
    en_to_fr['Nigeria'] = 'nigeria'
    fr_to_vec['nigeria'] = fr_embeddings['nigeria']
    en_to_fr['Mali'] = 'mali'
    fr_to_vec['mali'] = fr_embeddings['mali']
    fr_to_vec['grece'] = fr_embeddings['grèce']
    en_to_fr['Armenia'] = 'arménie'
    fr_to_vec['arménie'] = fr_embeddings['arménie']
    en_to_fr['Ecuador'] = 'ecuador'
    fr_to_vec['ecuador'] = fr_embeddings['ecuador']
    en_to_fr['Niger'] = 'niger'
    fr_to_vec['niger'] = fr_embeddings['niger']
    return en_to_fr, fr_to_vec

def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()
