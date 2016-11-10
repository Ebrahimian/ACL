import time
import sys
import os
import random
import numpy
from load import ds_load
from accuracy import accuracy
from elman import model
from tools import shuffle, minibatch, contextwin


if __name__ == '__main__':

    s = {'fold': 3,  # 5 folds 0,1,2,3,4
         'lr': 0.0627142536696559,
         'verbose': 1,
         'decay': False,  # decay on the learning rate if improvement stops
         'win': 7,  # number of words in the context window
         'bs': 9,  # number of backprop through time steps
         'nhidden': 100,  # number of hidden units
         'seed': 345,
         'emb_dimension': 100,  # dimension of word embedding
         'nepochs': 50}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, test_set, dicts = ds_load()
    ind2lemmas_tr, ind2labels = dicts['ind2lemmas_tr'], dicts['ind2labels']

    train_lex, train_y = train_set
    test_lex, test_y = train_set
    valid_lex, valid_y = train_set

    vocsize = len(ind2lemmas_tr)
    nclasses = len(ind2labels)
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(nh=s['nhidden'],
                nc=nclasses,
                ne=vocsize,
                de=s['emb_dimension'],
                cs=s['win'])

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], s['win'])
            words = map(lambda x: numpy.asarray(x).astype('int32'), \
                        minibatch(cwords, s['bs']))
            labels = train_y[i]
            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%' % (
                    e, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: ind2labels[x], \
                                rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) \
                            for x in test_lex]
        groundtruth_test = [map(lambda x: ind2labels[x], y) for y in test_y]
        words_test = [map(lambda x: ind2lemmas_tr[x], w) for w in test_lex]

        predictions_valid = [map(lambda x: ind2labels[x], \
                                 rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) \
                             for x in valid_lex]
        groundtruth_valid = [map(lambda x: ind2labels[x], y) for y in valid_y]
        words_valid = [map(lambda x: ind2lemmas_tr[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl


        res_test = accuracy(predictions_test, groundtruth_test, words_test)
        res_valid = accuracy(predictions_valid, groundtruth_valid, words_valid)

        if res_valid > best_f1:
            rnn.save(folder)
            best_f1 = res_valid

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be'] - s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', best_f1

