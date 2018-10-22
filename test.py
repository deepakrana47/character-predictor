from char_pridect import PredictWord
import sys

inp = "The world are the dangr"

pword = PredictWord(model_file="model_file.save")
while 1:
    inp=input("> ")
    if inp == 'q' or inp == 'Q':
        break
    elif inp == '':
        continue
    sys.stdout.write(str(pword.predict(inp))+'\n')