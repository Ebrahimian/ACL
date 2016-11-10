"""
Verbal MWEs include
idioms (*let the cat out of the bag*),
light verb constructions (*make a decision*),
verb-particle constructions (*give up*), and
inherently reflexive verbs (*se suicider* 'to suicide' in French).
"""

from patternExtractor import *



### Pattern Extraction
def pattern_extractor():
    """
    :return: avilable mwes and possible POS patterns in training data
    """
    train_sentences = file2sentencs('train.txt')
    mwes = []
    pos_patters = []
    for i in train_sentences:
        sentence_String = ' '.join(x[1] for x in train_sentences[i])
        sentence_String_tokenized = nltk.word_tokenize(sentence_String)
        pos_list = nltk.pos_tag(sentence_String_tokenized)
        sentences_pos = pos_append(train_sentences[i], pos_list)
        sentences_lemma = lema_append(sentences_pos)

        lables = get_lable(train_sentences[i])
        populated_sentences = lable_populate(sentences_lemma, lables)

        mwe_extract_string = mwe_extract(populated_sentences)
        Mwe_type = ['ID', 'LVC', 'VPC', 'OTH']
        numOfMwe = ['1', '2', '3', '4', '5']

        for n in Mwe_type:
            for m in numOfMwe:
                string = ' '.join([x[1] for x in mwe_extract_string if x[2] == n and x[0] == m])
                pos_pattern = '+'.join([get_simple_pos(x[3]) for x in mwe_extract_string if x[2] == n and x[0] == m])

                if string:
                    mwes.append((string, n, pos_pattern))
    for i in pos_pattern_extractor(mwes):
        pos_patters.append(i)
    return mwes, pos_patters


def base_line_evaluation():
    """
    :return: naive pattern matching ( base line evaluation)
    """
    mwes, pos_patters = pattern_extractor()
    test_sentences = file2sentencs('test.txt')

    for i in test_sentences:
        sentence_String = ' '.join(x[1] for x in test_sentences[i])
        sentence_String_tokenized = nltk.word_tokenize(sentence_String)
        pos_list = nltk.pos_tag(sentence_String_tokenized)
        sentences_pos = pos_append(test_sentences[i], pos_list)
        sentences_lemma = lema_append(sentences_pos)
        sentence_list_lemma = [x[11] for x in sentences_lemma]

        mwe_id = 1

        for mwe in mwes:
            token_ids = mwe_exact_match(mwe[0], sentence_list_lemma)
            if len(token_ids) == len(mwe[0].split()):
                for posit in token_ids:
                    if not isinstance(sentences_lemma[posit][4], int):
                        id1 = 4
                        id2 = 5
                    elif not isinstance(sentences_lemma[posit][6], int):
                        id1 = 6
                        id2 = 7
                        break
                    elif not isinstance(sentences_lemma[posit][8], int):
                        id1 = 8
                        id2 = 9
                        break
                for tn, token_id in enumerate(token_ids):
                    sentences_lemma[token_id][id1] = mwe_id
                    if tn == 0:
                        sentences_lemma[token_id][id2] = mwe[1]
                mwe_id += 1

        for i in sentences_lemma:
            print i[:10]
        print '*' * 50


### EVALUATION
if __name__ == '__main__':
    base_line_evaluation()











