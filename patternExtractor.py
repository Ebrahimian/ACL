from collections import defaultdict
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lmtzr = WordNetLemmatizer()


def unify_length(list_in, size):
    """
    :param list_in: sentences
    :param size: fixing size
    :return: all arrays extended to have the same size for ML task
    """
    if len(list_in) < size:
        list_in.extend([''] * (size - len(list_in)))
    return list_in

def get_lable(list_of_list):
    """
    :param list_of_list: each list is a token in a sentence with its features as the elements of the arays
    :return: available mwe labels in each sentence
    """
    labls = []
    mwe_positions = [4, 6, 8]
    for ls in list_of_list:
        for position in mwe_positions:
            if ls[position].isdigit() and ls[position + 1].isalpha():
                labls.append((ls[position], ls[position + 1]))

    return list(set(labls))

def lable_populate(sentence, lbls):
    """
    :param sentence: a list of list; each list is a token in a sentence with its features as the elements of the arays
    :param lbls: avilable lables in the sentence
    :return: adding lable string to each token which has the same label code
    """
    mwe_positions = [4, 6, 8]
    for token in sentence:
        for position in mwe_positions:
            for lable in lbls:
                if token[position] == lable[0]:
                    token[position + 1] = lable[1]

    return sentence

def get_wordnet_pos(treebank_tag):
    """
    :param treebank_tag:NLTK POS tags
    :return: mapping between NLTK tags and wordnet standard tags for Lemma extraction
    """

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def get_simple_pos(treebank_tag):
    """
    :param treebank_tag:NLTK POS tags
    :return: mapping between NLTK tags and simple tags for pos patter extraction
    """

    if treebank_tag.startswith('J'):
        return 'J'
    elif treebank_tag.startswith('V'):
        return 'V'
    elif treebank_tag.startswith('N'):
        return 'N'
    elif treebank_tag.startswith('R'):
        return 'R'
    else:
        return 'N'

def file2sentencs(fil):
    """
    :return: Dictionary mapping same-sized sentences to an index for recognizing sentence boundry.
    """
    with open(fil, 'r') as fl:
        sent = defaultdict(list)
        sentence_index = 0
        for line in fl:
            if line.isspace():
                sentence_index += 1
            else:
                line = line.rstrip().split('\t')
                line = unify_length(line, 12)
                if not line[0].isdigit():  # removing multi-tokens(do not -> don't)
                    continue
                sent[sentence_index].append(line)
    return sent

def pos_append(sentence, poslist):
    """
    :param sentence: a list of list; each list is a token in a sentence with its features as the elements of the arays
    :param poslist: POS tags tgged by NLTK
    :return: the same array of tokens with added POS at the end for each token
    """
    for token in sentence:
        pos = [pl[1] for pl in poslist if pl[0] == token[1]]
        if len(pos) > 0:
            token[10] = pos[0]
        else:  # making sure all tokens has the same length
            token[10] = ''
    return sentence

def lema_append(sentence):
    """
    :param sentence: a list of list; each list is a token in a sentence with its features as the elements of the arays
    :return: the same array of tokens with added POS at the end for each token
    """
    for token in sentence:
        lemma = lmtzr.lemmatize(word=token[1], pos=get_wordnet_pos(token[10]))
        token[11] = lemma.strip()
    return sentence

def mwe_extract(sentence):
    mwe_tokens = []
    mwe_positions = [4, 6, 8]
    for token in sentence:
        for mwe_position in mwe_positions:
            if token[mwe_position].isdigit():
                mwe_tokens.append((token[mwe_position], token[11], token[mwe_position + 1], token[10]))
    return mwe_tokens

def mwe_exact_match(mwe_pattern, string):
    """
    :param mwe_pattern: MWE extracted pattern
    :param string: sentence in which the pattern should be matched
    :return: token indices matched in ascending order
    """
    indices = []

    def string_match(mwe_token, strng):
        if mwe_token in strng:
            return strng.index(mwe_token)

    start_point = 0
    for i in mwe_pattern.split():
        st = string_match(i, string[start_point:len(string)])
        if st is None:
            break
        else:
            start_point += st
            indices.append(start_point)
    return indices

def pos_pattern_extractor(mwes):
    pos_pattern = []
    for i in mwes:
        pos_pattern.append((i[2], i[1]))
    return pos_pattern

def pos_extract_NN(sent_seq_classified):
    mwe_NN = []
    return mwe_NN

def pos_extract_VNN(sent_seq_classified):
    mwe_VNN = []
    return mwe_VNN

def pos_extract_VN(sent_seq_classified):
    mwe_VN = []
    return mwe_VN

def pos_extract_NV(sent_seq_classified):
    mwe_NV = []
    return mwe_NV