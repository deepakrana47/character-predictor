import numpy as np

def init_weight(Mi, Mo):
    return np.random.rand(Mi,Mo)/np.sqrt(Mi+Mo)