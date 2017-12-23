from char_pridect import predict_words, get_top_words
import numpy as np, sys, os
from GRU import gru
from vanilla_rnn import vrnn
from rnn_interface import train, test

def usage():
    print "Usage : python main.py [options]"
    print "options:"
    print "\t -rnn vrnn or gru: define type recurrent model use (by default 'gru')"
    print "\t -text file name: input file name use for trainning model for character prediction"
    print "\t -npredict number of pridection: given npredict number of words predicted"
    exit()

def parse_argv(argv):
    rnn = ''; ftext=''; npredict=10
    if len(argv) > 1:
        if argv[1] == '-help' or argv[1] == '--help' or argv[1] == '-h':
            usage()
        if '-rnn' in argv:
            ind=argv.index('-rnn')
            rnn = sys.argv[ind+1]
        if '-text' in argv:
            ind=argv.index('-text')
            ftext = sys.argv[ind+1]
        if '-npredict' in sys.argv:
            ind=sys.argv.index('-npredict')
            npredict = int(sys.argv[ind+1])
        if not os.path.isfile(ftext):
            print 'Enter a correct file path !!'
            exit()

    rnn = vrnn if rnn == 'vrnn' else gru
    fname = ftext if ftext else 'pg.txt'
    return rnn, fname, npredict

if __name__ == "__main__":

    rnn, fname, npredict = parse_argv(sys.argv)

    chars = list(set(open(fname, 'r').read()))
    i2c_map = {i: chars[i] for i in range(len(chars))}
    c2i_map = {chars[i]: i for i in range(len(chars))}
    v_size = len(chars)

    map_vect = {}
    for i in range(len(chars)):
        map_vect[chars[i]] = np.zeros((v_size, 1))
        map_vect[chars[i]][i] = 1.0

    # training recurrent model
    if os.path.isfile('./weights.pickle'):
        model = rnn(v_size, 250, v_size, wb='./weights.pickle')
    else:
        model = train(fname, rnn, map_vect)

    # testing model accuracy
    test = 0
    if test == 1:
        test(fname, model, map_vect)

    # console for word pridection
    print "Enter partial sentence after '>'"
    while True:
        text = raw_input('>').strip(' |\n|\t')
        if text == 'q' or text == 'Q':
            exit()
        elif text == '' or text == '\n':
            continue
        ind = 1
        itext = text[-25:]
        words = []
        count = 0
        words = predict_words(itext, model, map_vect, i2c_map, count)
        twords = get_top_words(words, npredict)
        print "partial sentence:",text
        print "Possible endings:"
        for word in twords:
            print '\t', word, '\t', text+word