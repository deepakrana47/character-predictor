import numpy as np
from Model import CharPredictNNModel

class PredictWord:

    def __init__(self, n_top=5):
        self.cmodel = CharPredictNNModel()
        self.cmodel.load("model_file.save")
        self.n = n_top

    def predict_chars(self, text):
        return self.cmodel.pridect(text)

    def get_top_words(self, words):
        a = [words[i] for i in words]
        a.sort(reverse=True)
        tword = []
        itr=0
        while itr<self.n and itr<len(a):
            for j in words:
                if words[j] == a[itr]:
                    tword.append(j)
            itr += 1
        return tword

    def predict_words(self, text, count):
        pchars = self.predict_chars(text)
        count += 1
        pwords = {}
        for char in pchars:
            if count >= 10:
                return {'':1.0}
            if char == ' ':
                return {' ':1.0}
            temp = self.predict_words(text[1:]+char, count)
            pwords.update({char+word:prob*pchars[char] for word, prob in temp.items()})
        return pwords

    def predict(self, text):
        words = self.predict_words(text, count=0)
        w1 = ''
        for i in reversed(text):
            if i == ' ':
                break
            w1 += i
        w2 = self.get_top_words(words)
        w1 = w1[::-1]
        return [w1+w for w in w2]
