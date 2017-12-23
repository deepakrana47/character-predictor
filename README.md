# character-predictor
A simple implementation for word completion based on next character pridection, where for character prediction a recurrent neural network is used. The recurrent network can be vanilla rnn or GRU rnn depend on given argument to program. 

Usage: python main.py [options]

    python main.py  or

    python main.py -rnn gru/vrnn -text text_file_name -npredict number_of_pridection

    [options]::

      -rnn vrnn/gru :: define type recurrent model use training (by default 'gru')"

      -text file name :: input file name use for trainning model for character prediction

      -npredict number of pridection :: given npredict number of words predicted
    
