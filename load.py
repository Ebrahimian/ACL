import cPickle

from patternExtractor import *


def input_generator(train_sentences):
    """
    :return: list of words(lemmas) and labels(1,-1)
    """
    sentences = []
    labels = []
    lemmas = []

    for i in train_sentences:
        sentence_String = ' '.join(x[1] for x in train_sentences[i])
        sentence_String_tokenized = nltk.word_tokenize(sentence_String)
        pos_list = nltk.pos_tag(sentence_String_tokenized)
        sentences_pos = pos_append(train_sentences[i], pos_list)
        sentences_lemma = lema_append(sentences_pos)
        sentences.append([x[11] for x in sentences_lemma])
        labels.append(['+' if x[4].isdigit() else '-' for x in sentences_lemma ])

        lemmas.extend([x[11] for x in sentences_lemma])
        lemmas = list(set(lemmas))

    label_ids = ['+', '-']
    sentences = [[lemmas.index(y) for y in x] for x in sentences]  # replacing lemmas in sentences with their ids
    labels = [[label_ids.index(y) for y in x] for x in labels] # replacing labels in sentences with their ids

    return sentences, labels, lemmas

def wrap_up():
    """
    :return: pkl object of the dataset
    """
    tr_sentences = file2sentencs('train.txt')
    ts_sentences = file2sentencs('test.txt')
    sentences_tr, labels_tr, ind2lemmas_tr = input_generator(tr_sentences)
    sentences_ts, labels_ts, ind2lemmas_ts = input_generator(ts_sentences)
    ind2labels = ['+', '-']

    dicts = {}
    dicts['ind2lemmas_tr'] = ind2lemmas_tr
    dicts['ind2labels'] = ind2labels
    train = (sentences_tr, labels_tr)
    test = (sentences_ts, labels_ts)

    pkl_object = (train, test, dicts)
    f = open('mwe_rnn.pkl', 'w')
    cPickle.dump(pkl_object, f)
    f.close()

def ds_load():
    train_set, test_set, dicts = cPickle.load(open('mwe_rnn.pkl'))
    return train_set, test_set, dicts

# if __name__ == '__main__':
#
#     ''' visualize a few sentences '''
#     train_set, test_set, dicts = cPickle.load(open('mwe_rnn.pkl'))
#
#     ind2lemmas_tr, ind2labels = dicts['ind2lemmas_tr'], dicts['ind2labels']
#
#
#     train_x,  train_y = train_set
#     test_x,  test_y  = test_set
#
#     print train_x[0]
#     print train_y[0]
#     print [ind2lemmas_tr[x] for x in train_x[0]]


