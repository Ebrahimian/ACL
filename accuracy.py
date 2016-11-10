import numpy
import pdb
import cPickle
import random
import os
import stat
import subprocess
from os.path import isfile, join
from os import chmod


def accuracy(p, g, w):
    """
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words
    """
    all_token = 0
    correct_token = 0
    assert len(p) == len(g)
    for sentence_id in range(len(p)):
        for token_id in range(len(p[sentence_id])):
            all_token +=1
            if p[sentence_id][token_id] == g[sentence_id][token_id]:
                correct_token +=1

    return (correct_token / all_token) * 100

