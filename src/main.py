import gensim
import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import utils


def NB(x_train, y_train, x_test, y_test):
    print('Naive Bayes...')

    clf = MultinomialNB(alpha=0.01)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    print('accuracy:' + str(accuracy))

    return accuracy


def KNN(x_train, y_train, x_test, y_test):
    print('KNN...')
    acc_list = []
    for x in range(1, 15):
        knnclf = KNeighborsClassifier(n_neighbors=x)
        knnclf.fit(x_train, y_train)
        preds = knnclf.predict(x_test)

        accuracy = accuracy_score(y_test, preds)
        print('K= ' + str(x) + ', accuracy:' + str(accuracy))
        acc_list.append(accuracy)

    return np.max(acc_list)


def SVM(x_train, y_train, x_test, y_test):
    print('SVM...')
    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    preds = preds.tolist()
    accuracy = accuracy_score(y_test, preds)
    print('accuracy:' + str(accuracy))

    return accuracy


def MLP(x_train, y_train, x_test, y_test):
    VALIDATION_SPLIT = 0.16
    TEST_SPLIT = 0.2

    all_texts = x_train + x_test
    all_labels = y_train + y_test

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')
    labels = to_categorical(np.asarray(all_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('(3) split data set...')
    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    p2 = int(len(data) * (1 - TEST_SPLIT))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: ' + str(len(x_train)))
    print('val docs: ' + str(len(x_val)))
    print('test docs: ' + str(len(x_test)))

    print('(5) training model...')

    model = Sequential()
    model.add(Dense(256, input_shape=(len(word_index) + 1,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.metrics_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save('mlp.h5')

    print('(6) testing model...')
    loss, accuracy = model.evaluate(x_test, y_test)
    print(str(loss), str(accuracy))

    return accuracy


def lstm(x_train, y_train, x_test, y_test):
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 200
    VALIDATION_SPLIT = 0.16
    TEST_SPLIT = 0.2

    all_texts = x_train + x_test
    all_labels = y_train + y_test

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(all_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('(3) split data set...')
    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    p2 = int(len(data) * (1 - TEST_SPLIT))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: ' + str(len(x_train)))
    print('val docs: ' + str(len(x_val)))
    print('test docs: ' + str(len(x_test)))

    print('(5) training model...')

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    print(model.metrics_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save('lstm.h5')

    print('(6) testing model...')
    loss, accuracy = model.evaluate(x_test, y_test)
    print(str(loss), str(accuracy))

    return accuracy


def word2vec_svm(x_train, y_train, x_test, y_test):
    VECTOR_DIR = '../pretrain/vectors.bin'
    EMBEDDING_DIM = 200

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    x_train = []
    x_test = []
    for train_doc in train_texts:
        words = train_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train.append(vector)
    for test_doc in test_texts:
        words = test_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_test.append(vector)
    print('train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0])))
    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))

    accuracy = SVM(x_train, y_train, x_test, y_test)

    return accuracy


def word2vec_lstm(x_train, y_train, x_test, y_test):
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 200
    VALIDATION_SPLIT = 0.16
    TEST_SPLIT = 0.2

    VECTOR_DIR = '../pretrain/vectors.bin'

    all_texts = x_train + x_test
    all_labels = y_train + y_test

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(all_labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print('(3) split data set...')
    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    p2 = int(len(data) * (1 - TEST_SPLIT))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: ' + str(len(x_train)))
    print('val docs: ' + str(len(x_val)))
    print('test docs: ' + str(len(x_test)))

    print('(4) load word2vec as embedding...')

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    not_in_model = 0
    in_model = 0
    for word, i in word_index.items():
        if word in w2v_model:
            in_model += 1
            embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
        else:
            not_in_model += 1
    print(str(not_in_model) + ' words not in w2v model')

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.metrics_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save('word_vector_lstm.h5')

    print('(6) testing model...')
    loss, accuracy = model.evaluate(x_test, y_test)
    print(str(loss), str(accuracy))

    return accuracy


if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels = utils.load_data()
    train_data, test_data = utils.tf_idf(train_texts, test_texts)
    x_train, y_train, x_test, y_test = train_data, train_labels, test_data, test_labels

    #  Naive Bayes
    nb_acc = NB(x_train, y_train, x_test, y_test)
    #
    # # KNN
    knn_acc = KNN(x_train, y_train, x_test, y_test)
    #
    # # SVM
    svm_acc = SVM(x_train, y_train, x_test, y_test)
    #
    # # MLP
    mlp_acc = MLP(train_texts, train_labels, test_texts, test_labels)
    #
    # # lstm
    lstm_acc = lstm(train_texts, train_labels, test_texts, test_labels)

    # # word2vec svm
    word2vec_svm_acc = word2vec_svm(train_texts, train_labels, test_texts, test_labels)

    # # word2vec lstm
    word2vec_lstm_acc = word2vec_lstm(train_texts, train_labels, test_texts, test_labels)

    print('=========================================================================')
    print(
        'Naive Bayes' + '  KNN  ' + '  SVM  ' + '  MLP  ' + '  lstm  ' + '  word2vec_svm  ' + '  word2vec_lstm  ')
    print("{:.4f}".format(nb_acc) + ' | ' + "{:.4f}".format(knn_acc) + ' | ' + "{:.4f}".format(svm_acc) + ' | ' + "{:.4f}".format(mlp_acc) + ' | ' + "{:.4f}".format(lstm_acc)
          + ' | ' + "{:.4f}".format(word2vec_svm_acc) + ' | ' + "{:.4f}".format(word2vec_lstm_acc))
    print('=========================================================================')
