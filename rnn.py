from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout


f = open('output_rnn.txt','a')
score_list = list()

for i in range(1,6):
    fileString = "imdb_master_"+str(i)+".csv"
    df_master = pd.read_csv(fileString, encoding='latin-1', index_col = 0)
    imdb_full = df_master[["review", "label"]][df_master.type.isin(['train'])].reset_index(drop=True)
    print("RUN: ",i)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    imdb_full = df_master[["review", "label"]][df_master.type.isin(['train'])].sample(frac=1).reset_index(drop=True)

    train_rev = pd.Series(imdb_full['review'][:12500])
    train_lab = pd.Series(imdb_full['label'][:12500])
    test_rev = pd.Series(imdb_full['review'][12500:])
    test_lab = pd.Series(imdb_full['label'][12500:])

    max_words = 500
    X_train = sequence.pad_sequences(train_rev, maxlen=max_words)
    X_test = sequence.pad_sequences(test_rev, maxlen=max_words)
    embedding_size=32
    model=Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    '''
    temp = train_rev
    train_rev = test_rev
    test_rev = temp

    temp = train_lab
    train_lab = test_lab
    test_lab = temp
    
    print("Swapping train and validation")
    print("---------------------------------------------------------------------------------")
    vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')
    vect_algo.fit(train_rev)
    Xf_train = vect_algo.transform(train_rev)
    Xf_test = vect_algo.transform(test_rev)
    '''
    print("DONE______________________________________________________________________________________")
'''
f.write("Logistic Regression- C-0.001 - TFIDF")
str1 = ' '.join(map(str,score_list))
f.write(str1)
f.write("\n")
'''
f.close()
    
