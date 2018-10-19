from char_pridect import PredictWord

text2 = "The world are the dan"

pword = PredictWord()
while 1:
    inp=input("> ")
    if inp == 'q' or inp == 'Q':
        break
    print(pword.predict(inp))