import numpy as np

def predict_chars(text, model, map_vect, i2c_map, n=2):
    x = [map_vect[i] for i in text]
    out = model.forward_pass(x)
    pred = np.argsort(out[-1], axis=0)[-n:,0].tolist()
    return {i2c_map[i]:out[-1][i] for i in pred}

def get_top_words(words, n):
    a = [words[i] for i in words]
    a.sort(reverse=True)
    tword = []
    itr=0
    while itr<n and itr<len(a):
        for j in words:
            if words[j] == a[itr]:
                tword.append(j)
        itr += 1
    return tword

def predict_words(text, model, map_vect, i2c_map, count):
    pchars = predict_chars(text, model, map_vect, i2c_map)
    count += 1
    pwords = {}
    for char in pchars:
        if count >= 10:
            return {'':1.0}
        if char == ' ':
            return {' ':1.0}
        temp = predict_words(text[1:]+char, model, map_vect, i2c_map, count)
        pwords.update({char+word:prob*pchars[char] for word, prob in temp.items()})
    return pwords