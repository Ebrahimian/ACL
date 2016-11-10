import itertools
from collections import Counter

import numpy as np

from patternExtractor import *


def mwe_extrator():
    """
    :return: liat of mwes with their labels
    """
    tr_sentences = file2sentencs('/ha/home/ebrahimian/Desktop/mwe/train.txt')

    mwes = []
    for i in tr_sentences:
        sentence_String = ' '.join(x[1] for x in tr_sentences[i])
        sentence_String_tokenized = nltk.word_tokenize(sentence_String)
        pos_list = nltk.pos_tag(sentence_String_tokenized)
        sentences_pos = pos_append(tr_sentences[i], pos_list)
        sentences_lemma = lema_append(sentences_pos)


        lables = get_lable(tr_sentences[i])
        populated_sentences = lable_populate(sentences_lemma, lables)

        mwe_extract_string = mwe_extract(populated_sentences)

        Mwe_type = ['ID', 'LVC', 'VPC', 'OTH']
        numOfMwe = ['1', '2', '3', '4', '5']

        for n in Mwe_type:
            for m in numOfMwe:
                string = ' '.join([x[1] for x in mwe_extract_string if x[2] == n and x[0] == m])
                if string:
                    mwes.append((string, n))
    return mwes
mwes = mwe_extrator()
# return ''.join([x.encode('utf-8') if isinstance(x, unicode) else str(x) for x in quest])
def load_data_and_labels():
    # Load data from main file

    # list of unique properties for indexing
    lables = list(set([x[1] for x in mwes]))
    n_classes = len(lables)

    # property to index
    q_lbl_index = [[x[0], lables.index(x[1])] for x in mwes]

    # index to one-hot representation
    q_lbl_one_hot = [[x[0], (np.arange(n_classes) == x[1]).astype(np.float32)] for x in q_lbl_index]

    # clear and split tokens in questions
    q_lbl_one_hot = [(x[0].split(), x[1]) for x in q_lbl_one_hot]
    x = [i[0] for i in q_lbl_one_hot]
    y = [j[1] for j in q_lbl_one_hot]

    return x, y, lables
# adding pad to fixate all question with the same size as the most lengthy sentences
def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    print 'sequence_length: ', sequence_length
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return sequence_length, padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index

    return vocabulary_inv

def build_input_data(sentences, labels, vocabulary_inv):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    vocabulary =  {x: i for i, x in enumerate(vocabulary_inv)}
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return x, y

def load_data():
    # Load and preprocess data
    x, y, lables = load_data_and_labels()
    sequence_length, sentences_padded = pad_sentences(x)
    vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, y, vocabulary_inv)
    return  x, y, vocabulary_inv, lables

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

