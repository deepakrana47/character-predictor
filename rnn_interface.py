import numpy as np, pickle

def sample_formation(text, seq_length, map_vect):
    samples = []
    t_size = len(text)
    for i in range(0, t_size - seq_length - 1):
        x = [map_vect[j] for j in text[i: i + seq_length]]
        y = [map_vect[j] for j in text[i + 1: i + seq_length + 1]]
        samples.append((x, y))
    return samples

def train(fname, rnn, map_vect):
    text = open(fname,'r').read()
    chars = list(set(text))
    v_size, t_size = len(chars), len(text)

    # recurrent NN initalization
    model = rnn(v_size, 250, v_size, optimize='rmsprop')

    # sample generation
    seq_length = 25
    samples = sample_formation(text, seq_length, map_vect)

    # RNN training parameter
    batch = 100
    miter = 20
    epoch0 = epoch = 60

    print "training start."
    while epoch > 0:
        itr = 0
        while itr < miter:
            deltaw = {}
            deltab= {}
            err = 0

            # mini_batch foramtion
            mini_batch = [samples[np.random.randint(0, len(samples))] for i in range(batch)]

            # mini_batch training
            while mini_batch:
                x,y = mini_batch.pop()
                model.forward_pass(x)
                dw, db, e = model.backward_pass(y)
                for j in dw:
                    if j in deltaw:
                        deltaw[j]+=dw[j]
                    else:
                        deltaw[j]=dw[j]
                for j in db:
                    if j in deltab:
                        deltab[j]+=db[j]
                    else:
                        deltab[j]=db[j]
                err += e

            # updating Recurrent network
            model.weight_update(model, {j:deltaw[j]/batch for j in deltaw}, {j:deltab[j]/batch for j in deltab}, neta=0.01)
            print '\t',itr,"batch error is",err/batch
            itr += 1

        print "\n %d epoch is completed\n" % (epoch0-epoch)
        epoch -= 1
    print "training complete."
    model.save_model('weights.pickle')
    return model

def test(fname, model, map_vect):
    text = open(fname, 'r').read()
    # sample generation
    seq_length = 25
    samples = sample_formation(text, seq_length, map_vect)

    # setting testing parameters
    iters = 1000
    correct = 0.0
    itr = 0

    # testing of RNN
    print "\ntesting start."
    while itr < iters:

        # selecting random sample from samples
        x, y = samples[np.random.randint(0, len(samples))]

        # producing output
        _o = model.forward_pass(x)
        if np.argmax(_o[-1]) == np.argmax(y[-1]):
            correct += 1
        itr += 1
    print "\ntesting complete.\n"
    print "correct:\t",correct
    print "incorrect:\t",iters-correct
    print "\naccuracy:\t",correct/iters
