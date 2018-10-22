# character-predictor
A python api for predicting the next word based on present text given. The character-predictor uses recurrent neural network(RNN) as prediction model for predicting the next character based on previous text.


## Requirements

Its a python implementaion compatible with both 2 and 3.

Requirements:

    sklearn <= 0.0
    numpy <= 1.15.0
    Theano <= 1.0.3

For adding to your project:
 Download the implementation to your project folder
 Add the lines to your file

    > from char_pridect import PredictWord
    > pword = PredictWord(model_file="model_file.save")

 Use **pword** for predicting word e.g.

    > text = "the partner of experi"
    > pword = pword.predict(inp)
    > pword
    ['experience ', 'experime ', 'experiencly ', 'experiend ', 'experiming ']

## Training

For training character-prediction RNN model to specific text corpus or data.

    if __name__ == '__main__':
        fname = 'pg.txt'
        cmodel = CharPredictNNModel(seq_len=32, hidden_lay_sz=(128,), model_file='model_file.save')
        cmodel.compile()
        cmodel.train(fname)

        # testing
        correct, total = cmodel.test(fname)
        sys.stdout.write("accuracy :", float(correct)/total)