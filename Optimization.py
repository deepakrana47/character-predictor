import theano, theano.tensor as T, numpy as np

def Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1) ** i_t
    fix2 = 1. - (1. - b2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def rmsprop(cost, params, lr=0.002, b1=0.1, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        m_t = b1 * m + (1-b1) * T.square(g)
        p_t = p - lr * g / (T.sqrt(m_t) + e)
        updates.append((m, m_t))
        updates.append((p, p_t))
    return updates